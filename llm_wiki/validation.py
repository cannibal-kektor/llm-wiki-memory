"""Ограничения ввода, строгий разбор JSON и проверка источника без модели и файлового ввода-вывода."""

import hashlib
import json
import re
from typing import Any
import unicodedata

from .domain import (
    ClaimCandidate, Source, SupportingQuotation, ValidationResult,
)

MAX_SOURCE_BYTES = 65_536
MAX_RESPONSE_BYTES = 262_144
MAX_CLAIMS = 32
MAX_STATEMENT_CHARS = 2_000
MAX_QUOTE_CHARS = 8_000
MAX_QUALIFIER_CHARS = 1_500
MAX_QUALIFIERS = 12
OPTIONAL_FIELDS = ("scope", "jurisdiction", "effective_date")
LIST_FIELDS = ("conditions", "exceptions")
CLAIM_FIELDS = frozenset(
    ("source_document_id", "statement", "quote", *LIST_FIELDS, *OPTIONAL_FIELDS)
)


def validate_document_id(document_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", document_id):
        raise ValueError("ID документа: 1–64 ASCII-буквы, цифры, '_' или '-'; первый символ — буква или цифра")


def has_unsafe_characters(text: str) -> bool:
    return any(
        unicodedata.category(char) in {"Cc", "Cf", "Cs"}
        and char not in "\n\r\t"
        for char in text
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON содержит повторяющиеся ключи объекта")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("JSON содержит неконечное число")


def parse_json(text: str) -> Any:
    """Отклонить JSON в Markdown, неполный, с повторными ключами, чрезмерной вложенностью или нарушением стандарта."""
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Ответ содержит некорректный Unicode") from None
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("Ответ превышает ограничение в байтах")
    try:
        return json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (json.JSONDecodeError, RecursionError):
        raise ValueError("Ответ содержит некорректный, обрезанный или чрезмерно вложенный JSON") from None


def _string(value: Any, limit: int, field: str) -> str:
    if (
        not isinstance(value, str) or not value.strip()
        or len(value) > limit or has_unsafe_characters(value)
    ):
        raise ValueError(f"{field}: требуется непустая безопасная строка длиной не более {limit} символов")
    return value


def _parse_batch(text: str) -> list[Any]:
    payload = parse_json(text)
    if not isinstance(payload, dict) or set(payload) != {"claims"}:
        raise ValueError("Ответ должен быть объектом с единственным полем 'claims'")
    claims = payload["claims"]
    if not isinstance(claims, list) or not 1 <= len(claims) <= MAX_CLAIMS:
        raise ValueError(f"claims должен быть списком из 1–{MAX_CLAIMS} кандидатов")
    return claims


def _qualifier_lists(item: dict[str, Any], prefix: str) -> dict[str, tuple[str, ...]]:
    qualifiers: dict[str, tuple[str, ...]] = {}
    for field in LIST_FIELDS:
        values = item[field]
        if not isinstance(values, list) or len(values) > MAX_QUALIFIERS:
            raise ValueError(f"{prefix}.{field}: требуется список не более чем из {MAX_QUALIFIERS} строк")
        qualifiers[field] = tuple(
            _string(value, MAX_QUALIFIER_CHARS, f"{prefix}.{field}")
            for value in values
        )
    return qualifiers


def _check_provenance(
    source: Source,
    reference: str,
    quote: str,
    qualifiers: dict[str, tuple[str, ...]],
    optional: dict[str, str | None],
) -> tuple[SupportingQuotation, tuple[str, ...]]:
    errors: list[str] = []
    if reference != source.identity.document_id:
        errors.append("source_document_id не совпадает с доверенным входным ID")
    start = source.text.find(quote)
    if start < 0:
        errors.append("Подтверждающая цитата не найдена дословно в исходном тексте")
    for field in LIST_FIELDS:
        if any(value not in quote for value in qualifiers[field]):
            errors.append(f"{field} содержит текст, отсутствующий в подтверждающей цитате")
    for field in OPTIONAL_FIELDS:
        if optional[field] is not None and optional[field] not in quote:
            errors.append(f"{field} отсутствует в подтверждающей цитате")
    quotation = SupportingQuotation(
        text=quote, start=None if start < 0 else start,
        end=None if start < 0 else start + len(quote),
    )
    return quotation, tuple(errors)


def _validate_candidate(item: Any, index: int, source: Source) -> ValidationResult:
    prefix = f"Кандидат {index}"
    if not isinstance(item, dict) or set(item) != CLAIM_FIELDS:
        raise ValueError(f"{prefix}: отсутствуют обязательные поля или присутствуют лишние")
    reference = _string(item["source_document_id"], 64, f"{prefix}.source_document_id")
    statement = _string(item["statement"], MAX_STATEMENT_CHARS, f"{prefix}.statement")
    quote = _string(item["quote"], MAX_QUOTE_CHARS, f"{prefix}.quote")
    qualifiers = _qualifier_lists(item, prefix)
    optional = {
        field: (None if item[field] is None
                else _string(item[field], MAX_QUALIFIER_CHARS, f"{prefix}.{field}"))
        for field in OPTIONAL_FIELDS
    }
    quotation, errors = _check_provenance(source, reference, quote, qualifiers, optional)
    # Это отпечаток извлечения, а не доказательство юридической эквивалентности или ID редакции.
    fingerprint = json.dumps(
        [source.identity.document_id, source.identity.content_sha256, item],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    candidate = ClaimCandidate(
        candidate_id="candidate-" + hashlib.sha256(fingerprint).hexdigest(),
        source=source.identity, statement=statement, quotation=quotation,
        conditions=qualifiers["conditions"], exceptions=qualifiers["exceptions"],
        scope=optional["scope"], jurisdiction=optional["jurisdiction"],
        effective_date=optional["effective_date"],
    )
    return ValidationResult(candidate, errors)


def validate_claims(text: str, source: Source) -> tuple[ValidationResult, ...]:
    """Отклонить пакет при ошибках схемы; сохранить ошибки происхождения каждого кандидата."""
    return tuple(
        _validate_candidate(item, index, source)
        for index, item in enumerate(_parse_batch(text), 1)
    )
