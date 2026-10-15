"""LE-3B.1: подготовка одного вызова Claim Construction — сверка запроса, журнал POST, оболочка ответа, replay.

Модуль сам ничего не отправляет: транспорт передаёт вызывающий код. HTTP допустим только в отдельно
разрешённом live-запуске; во всех остальных случаях — replay из частного файла обмена без сети.
LangExtract на уровне модуля не импортируется (класс ошибки транспорта LE-1 загружается лениво).
"""

import base64
import binascii
from collections.abc import Callable
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time
from typing import Any

from llm_wiki.llm import completion_content
from llm_wiki.structure import REPOSITORY
from llm_wiki.validation import MAX_RESPONSE_BYTES, _reject_constant, _unique_object, parse_json

from . import claims

EXCHANGE_CONTRACT = "llm-wiki-memory/le3b-claim-exchange/v1"
RUN_CONTRACT = "llm-wiki-memory/le3b-claim-run/v1"
# Один предел для оболочки Chat Completions: столько же принимает parse_json; reasoning_content входит в него.
MAX_ENVELOPE_BYTES = MAX_RESPONSE_BYTES
MAX_LEDGER_BYTES = 1_048_576
# Файл обмена хранит в base64 всё, что прочитал транспорт LE-1 (до 1 МиБ + 1 байт), в том числе ответ сверх предела.
MAX_EXCHANGE_BYTES = 1_500_000
BODY_FIELDS = frozenset(("messages", "stream", "max_tokens", "response_format"))
EXCHANGE_FIELDS = frozenset(("contract", "context_package_id", "request_body_sha256", "payload_sha256", "model",
                             "transport_outcome", "http_status", "response_b64"))
FINISH_REASONS = frozenset(("stop", "length", "content_filter", "tool_calls", "function_call"))
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")

Transport = Callable[[str, dict[str, Any], float], tuple[int, bytes]]


class RunError(Exception):
    """Отказ до отправки или в разборе частных файлов; сообщение не содержит текста запроса или ответа."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- запрос ----------------------------------------------------------------------------------------

def check_request(recorded: Any, regenerated: dict[str, Any]) -> None:
    """Запрос с диска должен совпасть с заново построенным целиком, а не только по body_sha256.

    Сравнение каноническим JSON: 4096.0 или true вместо 4096 и 1 считаются расхождением.
    """
    if claims._digest(recorded) != claims._digest(regenerated):
        raise RunError("Файл запроса не совпадает с запросом, заново построенным из пакета и снимка")


def payload(request: dict[str, Any], model: str) -> dict[str, Any]:
    """Тело POST: ровно два сообщения, stream=false, json_object, max_tokens реализации и одна модель."""
    body = request["body"]
    messages = body.get("messages") if isinstance(body, dict) else None
    if (not isinstance(body, dict) or set(body) != BODY_FIELDS or not isinstance(messages, list)
            or [message.get("role") if isinstance(message, dict) else None for message in messages] != ["system", "user"]
            or any(set(message) != {"role", "content"} or not isinstance(message["content"], str) for message in messages)
            or body["stream"] is not False or body["response_format"] != {"type": "json_object"}
            or type(body["max_tokens"]) is not int or body["max_tokens"] != claims.MAX_TOKENS):
        raise RunError("Тело запроса не соответствует закреплённой форме одного вызова Claim Construction")
    if not isinstance(model, str) or not model.strip():
        raise RunError("Нужен ровно один непустой идентификатор модели")
    return {"model": model, **body}


def payload_bytes(value: dict[str, Any]) -> bytes:
    """Та же сериализация, что в http_transport LE-1: по этим байтам считается payload_sha256."""
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


# --- журнал POST -----------------------------------------------------------------------------------

def ledger_path(raw: str) -> Path:
    """Журнал — существующий обычный файл внутри runtime, игнорируемый Git; символические ссылки не принимаются."""
    path = Path(raw)
    if path.is_symlink() or not path.is_file():
        raise RunError("Журнал POST должен быть существующим обычным файлом (не символической ссылкой)")
    resolved = path.resolve()
    if not resolved.is_relative_to(REPOSITORY / "runtime"):
        raise RunError("Журнал POST должен лежать внутри runtime")
    if subprocess.run(["git", "check-ignore", "-q", "--", str(resolved)], cwd=REPOSITORY,
                      capture_output=True).returncode != 0:
        raise RunError("Журнал POST должен игнорироваться Git")
    return resolved


def ledger_count(data: bytes) -> int:
    """Строгий разбор: первая строка baseline с целым count ≥ 0, далее только post; хвост без '\\n' — отказ.

    Каждая строка — строгий JSON: повторные ключи (на любой глубине) и NaN/Infinity — отказ, иначе
    {"count": 9, "count": 0} молча дал бы последнее значение и уменьшил учтённое число POST.
    """
    if not data or not data.endswith(b"\n"):
        raise RunError("Журнал POST пуст или содержит незавершённую строку (возможна прерванная запись)")
    try:
        lines = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
                 for line in data.decode("utf-8").split("\n")[:-1]]
    except (ValueError, RecursionError):  # включая UnicodeDecodeError, JSONDecodeError и повторные ключи
        raise RunError("Журнал POST содержит некорректную или неоднозначную строку") from None
    if not lines or not all(isinstance(entry, dict) for entry in lines):
        raise RunError("Журнал POST содержит некорректную строку")
    baseline = lines[0]
    if (baseline.get("kind") != "baseline" or type(baseline.get("count")) is not int or baseline["count"] < 0
            or any(entry.get("kind") != "post" for entry in lines[1:])):
        raise RunError("Журнал POST: нужна базовая строка baseline с целым count и далее только строки post")
    return baseline["count"] + len(lines) - 1


def _open_ledger(path: Path, flags: int) -> int:
    try:
        descriptor = os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        raise RunError("Журнал POST не открывается (символическая ссылка или нет доступа)") from None
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.geteuid():
        os.close(descriptor)
        raise RunError("Журнал POST должен быть обычным файлом владельца с правами 0600")
    return descriptor


def _read_all(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks, size = [], 0
    while chunk := os.read(descriptor, 65_536):
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_LEDGER_BYTES:
            raise RunError("Журнал POST слишком велик")
    return b"".join(chunks)


def read_ledger(path: Path) -> int:
    """Текущее число POST по журналу (только чтение, под разделяемой блокировкой)."""
    descriptor = _open_ledger(path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        return ledger_count(_read_all(descriptor))
    finally:
        os.close(descriptor)


def reserve_post(path: Path, cap: int, expected_count: int, entry: dict[str, Any]) -> tuple[int, int]:
    """Записать строку post до отправки под исключительной блокировкой: проверка и запись атомарны.

    Блокировка flock освобождается до отправки: следующий процесс уже увидит эту строку. Тайм-аут,
    обрыв или аварийное завершение после записи расходуют бюджет. Ошибка записи — отказ без отправки;
    частично записанная строка делает журнал непригодным (последующий разбор отказывает).
    """
    descriptor = _open_ledger(path, os.O_RDWR | os.O_APPEND)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        count = ledger_count(_read_all(descriptor))
        if count != expected_count:
            raise RunError(f"Журнал POST изменился: {count}, ожидалось {expected_count}; отправки не было")
        if count + 1 > cap:
            raise RunError(f"Бюджет POST исчерпан: {count} из {cap}; отправки не было")
        line = (json.dumps({"kind": "post", **entry}, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            written = os.write(descriptor, line)
            if written != len(line):
                raise OSError("partial write")
            os.fsync(descriptor)
        except OSError:
            raise RunError("Не удалось записать строку в журнал POST; отправки не было") from None
        return count, count + 1
    finally:
        os.close(descriptor)  # закрытие снимает flock


# --- отправка и оболочка ответа ---------------------------------------------------------------------

def send_once(send: Transport, url: str, value: dict[str, Any], timeout: float,
              reserve: Callable[[], tuple[int, int]] | None) -> dict[str, Any]:
    """Ровно одна попытка без повтора; reserve (журнал) выполняется до отправки и при отказе её отменяет."""
    from experiments.langextract.model import TransportError  # ленивый импорт: модуль LE-1 загружает LangExtract

    ledger = reserve() if reserve is not None else None
    attempt: dict[str, Any] = {"ledger_before_after": list(ledger) if ledger else None,
                               "budget_consumed": ledger is not None, "retry": "NONE"}
    started = time.monotonic()
    try:
        status, raw = send(url, value, timeout)
        attempt.update(transport_outcome="ok", http_status=status, raw=raw)
    except TransportError as error:
        attempt.update(transport_outcome=error.kind, http_status=error.status, raw=None)
    except Exception as error:  # подробности не сохраняются: сообщение может содержать данные
        attempt.update(transport_outcome=f"unexpected_{type(error).__name__}", http_status=None, raw=None)
    attempt["duration_s"] = round(time.monotonic() - started, 3)
    return attempt


def envelope(raw: bytes) -> tuple[str | None, dict[str, Any], str | None]:
    """Проверить оболочку Chat Completions: (content | None, безопасная диагностика, причина отказа | None).

    Решение о приёме — completion_content из llm_wiki.llm (ровно один choice, assistant, stop, без отказа и
    инструментов, непустой content); диагностика содержит только коды, числа и логические значения.
    """
    diagnostics: dict[str, Any] = {"response_bytes": len(raw), "response_sha256": sha256(raw)}
    if len(raw) > MAX_ENVELOPE_BYTES:
        return None, diagnostics, f"Ответ сервиса больше {MAX_ENVELOPE_BYTES} байт"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, diagnostics, "Ответ сервиса не соответствует UTF-8"
    try:
        document = parse_json(text)
    except ValueError as error:
        return None, diagnostics, str(error)
    if isinstance(document, dict):
        choices = document.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
        reason = choice.get("finish_reason")  # любой тип JSON; членство в множестве — только для строки
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        usage = document.get("usage") if isinstance(document.get("usage"), dict) else {}
        diagnostics.update(
            choices=len(choices) if isinstance(choices, list) else None,
            finish_reason=(reason if isinstance(reason, str) and reason in FINISH_REASONS
                           else None if reason is None else "OTHER"),
            usage={key: usage[key] for key in USAGE_FIELDS if type(usage.get(key)) is int},
            reasoning_chars=len(message["reasoning_content"]) if isinstance(message.get("reasoning_content"), str) else None,
            content_chars=len(message["content"]) if isinstance(message.get("content"), str) else None,
        )
    try:
        content = completion_content(document)
    except ValueError as error:  # сообщения completion_content фиксированы и не содержат текста ответа
        return None, diagnostics, str(error)
    diagnostics["content_sha256"] = sha256(content.encode("utf-8"))
    return content, diagnostics, None


# --- файл обмена (частный, для аудита и replay) -----------------------------------------------------

def exchange_record(context_package_id: str, body_sha256: str, payload_sha256: str, model: str,
                    attempt: dict[str, Any]) -> dict[str, Any]:
    """Полный ответ сервиса (байты в base64) и привязка к запросу; хранится только в частном каталоге."""
    raw = attempt["raw"]
    return {"contract": EXCHANGE_CONTRACT, "context_package_id": context_package_id,
            "request_body_sha256": body_sha256, "payload_sha256": payload_sha256, "model": model,
            "transport_outcome": attempt["transport_outcome"], "http_status": attempt["http_status"],
            "response_b64": base64.b64encode(raw).decode("ascii") if raw is not None else None}


def load_exchange(data: bytes) -> dict[str, Any]:
    """Строгий JSON (без повторных ключей и NaN) с собственным пределом размера: ответ внутри — в base64."""
    if len(data) > MAX_EXCHANGE_BYTES:
        raise RunError("Файл обмена слишком велик")
    try:
        record = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise RunError("Файл обмена не является строгим JSON") from None
    if not isinstance(record, dict) or set(record) != EXCHANGE_FIELDS or record["contract"] != EXCHANGE_CONTRACT:
        raise RunError("Файл обмена: неизвестный контракт или неожиданный набор полей")
    if not all(isinstance(record[key], str) for key in ("context_package_id", "request_body_sha256",
                                                        "payload_sha256", "model", "transport_outcome")):
        raise RunError("Файл обмена: поля привязки должны быть строками")
    status = record["http_status"]
    if (status is not None and type(status) is not int) or (record["transport_outcome"] == "ok" and status is None):
        raise RunError("Файл обмена: http_status — целое число (обязательно при transport_outcome = ok) или null")
    if (record["response_b64"] is None) != (record["transport_outcome"] != "ok"):
        raise RunError("Файл обмена: тело ответа есть тогда и только тогда, когда transport_outcome = ok")
    if record["response_b64"] is not None:
        try:
            base64.b64decode(record["response_b64"], validate=True)
        except (binascii.Error, ValueError, TypeError):
            raise RunError("Файл обмена: response_b64 не является base64") from None
    return record


def replay_transport(record: dict[str, Any], context_package_id: str, body_sha256: str) -> Transport:
    """Один ответ из файла обмена без сети; сверяет привязку к этому пакету и точным байтам запроса.

    Файл обмена не аутентифицирован: подделанный, но внутренне согласованный файл будет воспроизведён.
    """
    from experiments.langextract.model import TransportError

    used = []

    def send(url: str, value: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        if used:
            raise TransportError("replay_exhausted")
        used.append(True)
        if (record["context_package_id"] != context_package_id or record["request_body_sha256"] != body_sha256
                or record["payload_sha256"] != sha256(payload_bytes(value)) or record["model"] != value.get("model")):
            raise TransportError("replay_binding_mismatch")
        if record["response_b64"] is None:
            raise TransportError(record["transport_outcome"], record["http_status"])
        return record["http_status"], base64.b64decode(record["response_b64"])

    return send
