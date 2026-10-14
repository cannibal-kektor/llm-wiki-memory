"""Один синхронный запрос Chat Completions к явно настроенной локальной модели."""

from dataclasses import dataclass, field
from http.client import HTTPException
from importlib.resources import files
import ipaddress
import json
import math
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, ProxyHandler, build_opener

from .domain import Source
from .validation import (
    MAX_CLAIMS, MAX_QUOTE_CHARS, MAX_QUALIFIER_CHARS, MAX_QUALIFIERS,
    MAX_RESPONSE_BYTES, MAX_STATEMENT_CHARS, has_unsafe_characters, parse_json,
)


@dataclass(frozen=True)
class LLMConfig:
    endpoint: str = field(repr=False)
    model: str = field(repr=False)
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 60.0
    response_format: str = "text"

    def validate(self) -> None:
        if not self.endpoint:
            raise ValueError("Укажите --endpoint или LLM_WIKI_ENDPOINT; адреса по умолчанию нет")
        try:
            url = urlsplit(self.endpoint)
            port = url.port
            address = ipaddress.ip_address(url.hostname or "")
        except ValueError:
            raise ValueError("Адрес API должен быть базовым HTTP(S) URL с явно указанным локальным/частным IP") from None
        private_networks = (
            ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"), ipaddress.ip_network("fc00::/7"),
        )
        if not (address.is_loopback or any(address in network for network in private_networks)):
            raise ValueError("Адрес API должен содержать loopback или частный IP; публичные адреса запрещены")
        if (
            url.scheme not in {"http", "https"} or url.username is not None
            or url.password is not None or url.query or url.fragment
            or (port is not None and port < 1)
            or any(ord(char) <= 32 or ord(char) >= 127 for char in self.endpoint)
            or "\\" in self.endpoint
        ):
            raise ValueError("Адрес API должен быть базовым HTTP(S) URL без учётных данных, параметров запроса и фрагмента")
        if not self.model or len(self.model) > 256 or has_unsafe_characters(self.model):
            raise ValueError("Укажите непустой идентификатор модели: не более 256 безопасных символов")
        if not self.model.strip():
            raise ValueError("Идентификатор модели не может состоять только из пробелов")
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 300:
            raise ValueError("Тайм-аут должен быть конечным числом от 1 до 300 секунд")
        if self.response_format not in {"text", "json_object"}:
            raise ValueError("Формат ответа должен быть 'text' или 'json_object'")
        if self.api_key is not None and (
            not self.api_key or len(self.api_key) > 4096
            or any(ord(char) < 33 or ord(char) > 126 for char in self.api_key)
        ):
            raise ValueError("LLM_WIKI_API_KEY должен быть непустым ASCII-токеном без пробелов")


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def extraction_messages(source: Source) -> list[dict[str, str]]:
    example = {
        "claims": [{
            "source_document_id": source.identity.document_id,
            "statement": "Одна атомарная норма с сохранёнными ограничениями.",
            "quote": "Точная непрерывная цитата из документа.",
            "conditions": [], "exceptions": [], "scope": None,
            "jurisdiction": None, "effective_date": None,
        }]
    }

    replacements = {
        "SCHEMA_EXAMPLE": json.dumps(example, ensure_ascii=False),
        "MAX_CLAIMS": str(MAX_CLAIMS),
        "MAX_STATEMENT_CHARS": str(MAX_STATEMENT_CHARS),
        "MAX_QUOTE_CHARS": str(MAX_QUOTE_CHARS),
        "MAX_QUALIFIER_CHARS": str(MAX_QUALIFIER_CHARS),
        "MAX_QUALIFIERS": str(MAX_QUALIFIERS),
    }
    try:
        template = files("llm_wiki").joinpath("prompts/legal_extraction_ru.txt").read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError):
        raise ValueError("Не удалось прочитать UTF-8 ресурс prompts/legal_extraction_ru.txt") from None
    markers = re.findall(r"<<([A-Z_]+)>>", template)
    remainder = re.sub(r"<<[A-Z_]+>>", "", template)
    if set(markers) != set(replacements) or "<<" in remainder or ">>" in remainder:
        raise ValueError("Некорректные или отсутствующие маркеры в шаблоне системного промпта")
    instruction = re.sub(r"<<([A-Z_]+)>>", lambda match: replacements[match[1]], template)

    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps({
            "source_document_id": source.identity.document_id,
            "untrusted_source_text": source.text,
        }, ensure_ascii=False)},
    ]


def completion_content(payload: Any) -> str:
    """Проверить оболочку непотокового ответа перед проверкой схемы извлечения."""
    if not isinstance(payload, dict):
        raise ValueError("Ответ сервиса должен быть JSON-объектом Chat Completions")
    if "error" in payload:
        raise ValueError("Сервис вернул ошибку API; подробности скрыты")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Ответ сервиса должен содержать ровно один вариант завершения")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Ответ модели обрезан; кандидаты не приняты")
    if choice.get("finish_reason") == "content_filter":
        raise ValueError("Модель отказала в ответе или ответ отфильтрован")
    if choice.get("finish_reason") != "stop":
        raise ValueError("Причина завершения отсутствует или не поддерживается; кандидаты не приняты")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("В ответе сервиса отсутствует сообщение assistant")
    if message.get("refusal"):
        raise ValueError("Модель отказалась выполнять извлечение; текст отказа скрыт")
    if message.get("tool_calls") or message.get("function_call"):
        raise ValueError("Модель запросила инструменты; их выполнение не поддерживается")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Модель вернула пустое или неподдерживаемое содержимое сообщения")
    return content


def request_extraction(source: Source, config: LLMConfig) -> str:
    config.validate()
    payload: dict[str, Any] = {
        "model": config.model, "messages": extraction_messages(source),
        "stream": False, "max_tokens": 8192,
    }
    if config.response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if config.api_key is not None:
        headers["Authorization"] = "Bearer " + config.api_key
    request = Request(
        config.endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST",
    )
    opener = build_opener(ProxyHandler({}), _NoRedirects())
    try:
        with opener.open(request, timeout=config.timeout) as response:
            if response.status != 200:
                raise ValueError("Сервис вернул неожиданный HTTP-статус")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        code = error.code
        error.close()
        if 300 <= code < 400:
            raise ValueError("Сервис перенаправил запрос; перенаправления запрещены") from None
        if code in {401, 403}:
            raise ValueError(f"Сервис HTTP {code}: учётные данные или доступ отклонены") from None
        if code in {400, 422} and config.response_format == "json_object":
            raise ValueError(
                f"Сервис HTTP {code}: запрос отклонён; возможно, json_object не поддерживается. "
                "При необходимости явно выберите 'text'; резервный запрос не отправлялся"
            ) from None
        raise ValueError(f"Сервис HTTP {code}: ошибка запроса; подробности скрыты, повтора нет") from None
    except TimeoutError:
        raise ValueError("Истёк тайм-аут запроса к локальной модели; повтора нет") from None
    except URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise ValueError("Истёк тайм-аут запроса к локальной модели; повтора нет") from None
        raise ValueError("Локальная модель недоступна или проверка TLS не пройдена; подробности скрыты, повтора нет") from None
    except (OSError, HTTPException):
        raise ValueError("Ошибка транспорта локальной модели или неполный ответ; повтора нет") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Ответ сервиса превышает ограничение в байтах")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Ответ сервиса не соответствует UTF-8") from None
    return completion_content(parse_json(text))
