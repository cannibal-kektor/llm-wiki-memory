"""Модель LangExtract через публичный интерфейс BaseLanguageModel: один POST на промпт, полный учёт вызовов.

Встроенный провайдер OpenAI в 1.7.1 создаёт клиент SDK со значениями по умолчанию
(2 скрытых повтора, тайм-аут 600 с, прокси из окружения) и принимает непустой ответ с
finish_reason=length. Поэтому для реальных вызовов используется этот адаптер, переданный в
lx.extract(model=...), без изменения внутренних объектов библиотеки.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import hashlib
from http.client import HTTPException
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from langextract.core import base_model, exceptions, types

from llm_wiki.llm import LLMConfig, _NoRedirects

MAX_RESPONSE_BYTES = 1_048_576
SYSTEM_MESSAGE = "You are a helpful assistant that responds in JSON format."  # как у встроенного провайдера OpenAI


class CallFailed(exceptions.InferenceRuntimeError):
    """Вызов модели не дал пригодного ответа; вид сбоя — в записи вызова."""


@dataclass
class CallRecord:
    index: int
    task: str
    prompt: str
    prompt_sha256: str
    request: dict[str, Any]
    ignored_kwargs: list[str]
    outcome: str = "pending"
    http_status: int | None = None
    duration_s: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    response_bytes: int | None = None
    response_body: str | None = None
    content: str | None = None
    reasoning_chars: int | None = None


# Транспорт возвращает (HTTP-статус, тело) либо возбуждает исключение с видом сбоя.
Transport = Callable[[str, dict[str, Any], float], tuple[int, bytes]]


class TransportError(Exception):
    def __init__(self, kind: str, status: int | None = None) -> None:
        super().__init__(kind)
        self.kind = kind
        self.status = status


def http_transport(config: LLMConfig) -> Transport:
    """Один синхронный POST без прокси, перенаправлений и повторов; подробности ошибок сервера не читаются."""
    opener = build_opener(ProxyHandler({}), _NoRedirects())

    def send(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if config.api_key is not None:
            headers["Authorization"] = "Bearer " + config.api_key
        request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            code = error.code
            error.close()
            kind = ("redirect_refused" if 300 <= code < 400
                    else "unauthorized" if code in {401, 403} else "http_error")
            raise TransportError(kind, code) from None
        except TimeoutError:
            raise TransportError("timeout") from None
        except URLError as error:
            raise TransportError("timeout" if isinstance(error.reason, TimeoutError) else "transport_error") from None
        except (OSError, HTTPException):
            raise TransportError("transport_error") from None

    return send


def replay_transport(recorded: Sequence[dict[str, Any]]) -> Transport:
    """Воспроизвести ранее записанные ответы по порядку, без сети."""
    queue = list(recorded)

    def send(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        if not queue:
            raise TransportError("replay_exhausted")
        item = queue.pop(0)
        if item.get("response_body") is None:
            raise TransportError(str(item.get("outcome") or "transport_error"), item.get("http_status"))
        status = item.get("http_status") or 200
        if status != 200:
            raise TransportError("http_error", status)
        return status, item["response_body"].encode("utf-8")

    return send


def _classify(record: CallRecord, raw: bytes) -> str:
    """Вид результата по оболочке Chat Completions; содержимое сохраняется и при отказе."""
    if len(raw) > MAX_RESPONSE_BYTES:
        return "response_too_large"
    try:
        body = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "invalid_utf8"
    record.response_body = body
    try:
        envelope = json.loads(body)
    except json.JSONDecodeError:
        return "invalid_envelope_json"
    if not isinstance(envelope, dict):
        return "invalid_envelope"
    if "error" in envelope:
        return "api_error"
    record.usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else None
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return "invalid_envelope"
    choice = choices[0]
    message = choice.get("message")
    record.finish_reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
    if not isinstance(message, dict):
        return "invalid_envelope"
    if isinstance(message.get("content"), str):
        record.content = message["content"]
    if isinstance(message.get("reasoning_content"), str):
        record.reasoning_chars = len(message["reasoning_content"])
    if record.finish_reason == "length":
        return "truncated"
    if record.finish_reason == "content_filter" or message.get("refusal"):
        return "refusal"
    if message.get("tool_calls") or message.get("function_call"):
        return "tool_calls"
    if record.finish_reason != "stop":
        return "unsupported_finish_reason"
    if not record.content or not record.content.strip():
        return "empty_content"
    return "ok"


class GuardedChatModel(base_model.BaseLanguageModel):
    """Последовательные вызовы Chat Completions с жёстким лимитом и записью каждого вызова."""

    def __init__(self, config: LLMConfig, transport: Transport, *, max_calls: int,
                 max_tokens: int, temperature: float | None) -> None:
        super().__init__()
        self.format_type = types.FormatType.JSON
        self._config = config
        self._transport = transport
        self._max_calls = max_calls
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.task = "unset"
        self.calls: list[CallRecord] = []

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Iterator[Sequence[types.ScoredOutput]]:
        # Annotator передаёт сюда параметры выравнивания; в запрос они не попадают.
        ignored = sorted(kwargs)
        for prompt in batch_prompts:
            yield [types.ScoredOutput(score=1.0, output=self._call(prompt, ignored))]

    def _call(self, prompt: str, ignored: list[str]) -> str:
        request: dict[str, Any] = {"model": self._config.model, "n": 1, "stream": False,
                                   "max_tokens": self._max_tokens}
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._config.response_format == "json_object":
            request["response_format"] = {"type": "json_object"}
        record = CallRecord(len(self.calls) + 1, self.task, prompt,
                            hashlib.sha256(prompt.encode("utf-8")).hexdigest(), request, ignored)
        self.calls.append(record)
        if record.index > self._max_calls:
            record.outcome = "budget_exceeded"
            raise CallFailed("Превышен лимит вызовов модели; запрос не отправлен")
        payload = {**request, "messages": [{"role": "system", "content": SYSTEM_MESSAGE},
                                           {"role": "user", "content": prompt}]}
        started = time.monotonic()
        try:
            record.http_status, raw = self._transport(
                self._config.endpoint.rstrip("/") + "/chat/completions", payload, self._config.timeout)
        except TransportError as error:
            record.outcome, record.http_status = error.kind, error.status
            raise CallFailed(f"Сбой вызова модели: {error.kind}; повтора нет") from None
        finally:
            record.duration_s = round(time.monotonic() - started, 3)
        record.response_bytes = len(raw)
        record.outcome = _classify(record, raw)
        if record.outcome != "ok":
            raise CallFailed(f"Ответ модели не принят: {record.outcome}; повтора нет")
        return record.content or ""


def call_summary(calls: Sequence[CallRecord]) -> list[dict[str, Any]]:
    """Сведения о вызовах без промптов и тел ответов (для консоли и краткой части отчёта)."""
    return [{
        "index": call.index, "task": call.task, "outcome": call.outcome, "http_status": call.http_status,
        "finish_reason": call.finish_reason, "duration_s": call.duration_s, "usage": call.usage,
        "prompt_chars": len(call.prompt), "response_bytes": call.response_bytes,
        "reasoning_chars": call.reasoning_chars,
    } for call in calls]


def to_dict(call: CallRecord) -> dict[str, Any]:
    return {key: getattr(call, key) for key in CallRecord.__dataclass_fields__}
