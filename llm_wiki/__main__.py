"""Прочитать один разрешённый источник и показать кандидатов без сохранения результатов."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from .domain import Source, SourceIdentity, ValidationResult
from .llm import LLMConfig, request_extraction
from .validation import (
    MAX_CLAIMS, MAX_SOURCE_BYTES, has_unsafe_characters, validate_claims, validate_document_id,
)


def read_source(path: Path, document_id: str) -> Source:
    validate_document_id(document_id)
    try:
        if not path.is_file():
            raise ValueError("Источник должен быть обычным текстовым файлом UTF-8")
        with path.open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Источник должен быть обычным текстовым файлом UTF-8")
            raw = stream.read(MAX_SOURCE_BYTES + 1)
    except OSError:
        raise ValueError("Не удалось прочитать файл; проверьте путь и права доступа") from None
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError(f"Размер источника превышает {MAX_SOURCE_BYTES} байт; обрезка не выполняется")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Источник должен быть текстом UTF-8") from None
    if not text.strip() or has_unsafe_characters(text):
        raise ValueError("Источник пуст или содержит неподдерживаемые управляющие символы")
    return Source(SourceIdentity(document_id, hashlib.sha256(raw).hexdigest()), text)


def print_preview(source: Source, results: tuple[ValidationResult, ...]) -> None:
    print("ПРЕДПРОСМОТР — НЕПРОВЕРЕННЫЕ КАНДИДАТЫ; публикация не выполняется")
    print("Проверка: ПРОЙДЕНА (PASS) означает только корректную структуру и дословное происхождение.\nЮридический смысл, полнота и официальный статус не подтверждены. Отсылки к недоступным актам не разрешены.")
    print(f"Документ-источник: {source.identity.document_id}")
    print(f"SHA-256 содержимого: {source.identity.content_sha256} (снимок входных байтов, не юридическая редакция)")
    print("Редакция источника и правовой раздел: неизвестны")
    print(f"Извлечено кандидатов: {len(results)}; предел: {MAX_CLAIMS}. Полнота охвата не гарантируется.")
    if len(results) == MAX_CLAIMS:
        print("ВНИМАНИЕ: достигнут предел кандидатов; часть положений могла остаться неизвлечённой.")
    for index, result in enumerate(results, 1):
        claim = result.candidate
        print(f"\nКандидат {index}: {claim.candidate_id}")
        print("Статус: НЕПРОВЕРЕННЫЙ / требуется проверка человеком")
        print("Утверждение: " + json.dumps(claim.statement, ensure_ascii=False))
        print(f"Ссылка: документ={claim.source.document_id}; редакция=неизвестна; раздел=неизвестен")
        quote = claim.quotation
        if quote.start is not None:
            print(f"Позиция цитаты: [{quote.start}, {quote.end}) — смещения символов Unicode от нуля")
        else:
            print("Позиция цитаты: недоступна (цитата не совпала)")
        print("Цитата из ответа модели (дословность проверяется ниже): " + json.dumps(quote.text, ensure_ascii=False))
        for field in ("conditions", "exceptions", "scope", "jurisdiction", "effective_date"):
            print(field + ": " + json.dumps(getattr(claim, field), ensure_ascii=False))
        if result.passed:
            print("Проверка: ПРОЙДЕНА (PASS) — только схема, ID источника, цитата и фрагменты уточнений")
        else:
            print("Проверка: НЕ ПРОЙДЕНА (FAIL) — " + "; ".join(result.errors))
    if not all(result.passed for result in results):
        print("\nПАКЕТ ОТКЛОНЁН — хотя бы один кандидат не прошёл проверку; весь пакет не принят")
    else:
        print("\nПАКЕТ ПРОШЁЛ ПРОВЕРКИ ИСТОЧНИКА — все кандидаты остаются НЕПРОВЕРЕННЫМИ")


class _RussianArgumentParser(argparse.ArgumentParser):
    """Локализованные ошибки CLI без вывода потенциально секретных аргументов."""

    def format_usage(self) -> str:
        return super().format_usage().replace("usage: ", "Использование: ", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage: ", "Использование: ", 1)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, "Ошибка аргументов: проверьте имена и значения параметров. "
                  "Обязательны input, --document-id, --approved-source и --dry-run. "
                  "Справка: --help.\n")


def main(argv: list[str] | None = None) -> int:
    parser = _RussianArgumentParser(
        add_help=False,
        description="Извлечение непроверенных кандидатов из одного полного TXT в UTF-8. "
        "--dry-run отправляет текст разрешённой локальной модели одним запросом; результаты не сохраняются и не публикуются."
    )
    parser._positionals.title = "Позиционные аргументы"
    parser._optionals.title = "Параметры"
    parser.add_argument("-h", "--help", action="help", help="показать справку и выйти")
    parser.add_argument("input", type=Path, help="путь к полному TXT в UTF-8")
    parser.add_argument("--document-id", required=True, help="доверенный ID источника (ASCII-буквы/цифры/_/-)")
    parser.add_argument("--dry-run", required=True, action="store_true", help="обязателен; только предпросмотр, с отправкой текста модели")
    parser.add_argument(
        "--approved-source", required=True, action="store_true",
        help="подтвердить разрешение отправить выбранный источник настроенному одобренному сервису модели; не подтверждает юридическую правильность или права на публикацию",
    )
    parser.add_argument("--endpoint", help="базовый URL одобренного локального API; переменная LLM_WIKI_ENDPOINT")
    parser.add_argument("--model", help="ID локальной модели; переменная LLM_WIKI_MODEL")
    parser.add_argument("--timeout", help="тайм-аут операции сокета 1–300 секунд; LLM_WIKI_TIMEOUT, по умолчанию 60")
    parser.add_argument(
        "--response-format", help="text (по умолчанию) или json_object; LLM_WIKI_RESPONSE_FORMAT",
    )
    args = parser.parse_args(argv)
    try:
        timeout_text = args.timeout if args.timeout is not None else os.environ.get("LLM_WIKI_TIMEOUT", "60")
        try:
            timeout = float(timeout_text)
        except ValueError:
            raise ValueError("Тайм-аут должен быть числом от 1 до 300 секунд") from None
        config = LLMConfig(
            endpoint=args.endpoint if args.endpoint is not None else os.environ.get("LLM_WIKI_ENDPOINT", ""),
            model=args.model if args.model is not None else os.environ.get("LLM_WIKI_MODEL", ""),
            api_key=os.environ.get("LLM_WIKI_API_KEY"), timeout=timeout,
            response_format=(args.response_format if args.response_format is not None
                             else os.environ.get("LLM_WIKI_RESPONSE_FORMAT", "text")),
        )
        config.validate()
        source = read_source(args.input, args.document_id)
        print("Предпросмотр: подготовка одного запроса к явно настроенной локальной модели.\nЦитаты попадут в вывод терминала; журналы сеанса могут содержать исходный текст.", file=sys.stderr)
        results = validate_claims(request_extraction(source, config), source)
        print_preview(source, results)
        return 0 if all(result.passed for result in results) else 1
    except ValueError as error:
        print(f"Ошибка: {error}. Пакет не принят; запись Wiki не выполнялась.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
