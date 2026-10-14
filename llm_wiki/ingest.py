"""Локальное структурное преобразование одного RTF/DOCX/PDF в снимок DoclingDocument.

Без LLM, Wiki, Git и сетевых обращений. Снимок — производное представление Docling,
а не юридический оригинал; его текстовая и структурная точность не проверяется.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import platform
import resource
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any
import warnings

from .validation import validate_document_id

FORMAT_BY_SUFFIX = {".rtf": "rtf", ".docx": "docx", ".pdf": "pdf"}
SIGNATURE_BY_FORMAT = {"rtf": b"{\\rtf", "docx": b"PK\x03\x04", "pdf": b"%PDF-"}
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
MAX_BYTES_LIMIT = 200 * 1024 * 1024
DEFAULT_MAX_PAGES = 300
MAX_PAGES_LIMIT = 2_000
LIBREOFFICE_PROBE_TIMEOUT_S = 30
MAX_DIAGNOSTICS = 50
MAX_DIAGNOSTIC_CHARS = 500
PDF_MODEL_MODULES = ("torch", "docling_ibm_models")
VERSIONED_PACKAGES = (
    "docling-slim", "docling-core", "docling-parse", "python-docx", "pypdfium2", "pydantic", "lxml",
)
# Защита в глубину для библиотек Hugging Face; сама по себе не запрещает сеть.
OFFLINE_ENVIRONMENT = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
REPORT_SCHEMA = "llm-wiki-memory/ingest-report/v1"
PROVENANCE_LIMITS = (
    "SHA-256 источника вычислен по исходным байтам файла; снимок — результат преобразования Docling, не эти байты.",
    "Позиции и порядок в снимке относятся к снимку, а не к байтам или символам оригинала.",
    "Редакция, юридический статус, даты действия и правовая иерархия не устанавливаются.",
    "Заголовки, нумерация, списки, таблицы, сноски и порядок чтения могут быть утрачены или искажены; требуется ручная проверка.",
)


@dataclass(frozen=True)
class SourceFile:
    filename: str
    input_format: str
    data: bytes
    sha256: str


@dataclass(frozen=True)
class OutputNames:
    snapshot: str
    incomplete_snapshot: str
    report: str

    @classmethod
    def for_document(cls, document_id: str) -> "OutputNames":
        return cls(
            snapshot=f"{document_id}.docling.json",
            incomplete_snapshot=f"{document_id}.INCOMPLETE.docling.json",
            report=f"{document_id}.report.json",
        )

    def all(self) -> tuple[str, ...]:
        return (self.snapshot, self.incomplete_snapshot, self.report)


class _DiagnosticsHandler(logging.Handler):
    """Сохраняет ограниченную диагностику Python logging уровня WARNING и выше."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.counts: Counter[str] = Counter()
        self.records: list[dict[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.counts[f"{record.name}:{record.levelname}"] += 1
        if len(self.records) < MAX_DIAGNOSTICS:
            self.records.append({
                "logger": record.name, "level": record.levelname,
                "message": _bounded(record.getMessage()),
            })


def _bounded(text: str) -> str:
    return text if len(text) <= MAX_DIAGNOSTIC_CHARS else text[:MAX_DIAGNOSTIC_CHARS] + "…[обрезано]"


def _bounded_int(text: str, flag: str, upper: int) -> int:
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{flag}: требуется целое число от 1 до {upper}") from None
    if not 1 <= value <= upper:
        raise ValueError(f"{flag}: требуется целое число от 1 до {upper}")
    return value


def read_source_file(path: Path, max_bytes: int) -> SourceFile:
    input_format = FORMAT_BY_SUFFIX.get(path.suffix.lower())
    if input_format is None:
        raise ValueError("Поддерживаются только .rtf, .docx и .pdf (регистр расширения не важен)")
    try:
        if not path.is_file():
            raise ValueError("Источник должен быть существующим обычным файлом")
        with path.open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Источник должен быть существующим обычным файлом")
            data = stream.read(max_bytes + 1)
    except OSError:
        raise ValueError("Не удалось прочитать файл; проверьте путь и права доступа") from None
    if not data:
        raise ValueError("Источник пуст")
    if len(data) > max_bytes:
        raise ValueError(f"Размер источника превышает --max-bytes={max_bytes}; обрезка не выполняется")
    if not data.startswith(SIGNATURE_BY_FORMAT[input_format]):
        raise ValueError(f"Сигнатура содержимого не соответствует формату {input_format.upper()}")
    return SourceFile(path.name, input_format, data, hashlib.sha256(data).hexdigest())


def require_network_isolation() -> None:
    """Отказать, если процессу видны сетевые интерфейсы кроме loopback."""
    try:
        names = sorted(name for _, name in socket.if_nameindex() if name != "lo")
    except OSError:
        raise ValueError("Не удалось проверить сетевые интерфейсы") from None
    if names:
        raise ValueError(
            f"Процессу доступны сетевые интерфейсы кроме loopback ({len(names)} шт.). "
            "Запустите в изолированном сетевом пространстве, например: unshare -rn python -m llm_wiki.ingest …"
        )


def resolve_libreoffice(explicit: Path | None) -> tuple[str, str]:
    """Найти и проверить локальный soffice; вернуть путь и строку версии."""
    candidate = str(explicit.resolve()) if explicit is not None else shutil.which("soffice")
    if not candidate or not Path(candidate).is_file() or not os.access(candidate, os.X_OK):
        raise ValueError(
            "Для RTF требуется локальный исполняемый LibreOffice (soffice); не найден. "
            "Установите его отдельно или укажите --libreoffice"
        )
    try:
        completed = subprocess.run(
            [candidate, "--version"], capture_output=True, text=True,
            timeout=LIBREOFFICE_PROBE_TIMEOUT_S, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        raise ValueError("LibreOffice не запустился с --version") from None
    lines = completed.stdout.strip().splitlines()
    if not lines or not lines[0].startswith("LibreOffice"):
        raise ValueError("Не удалось определить версию LibreOffice")
    return candidate, _bounded(lines[0])


def check_pdf_prerequisites(artifacts_path: Path | None) -> None:
    if artifacts_path is None:
        raise ValueError(
            "Для PDF обязателен --artifacts-path: локальный каталог заранее полученных моделей Docling; "
            "скачивание не выполняется"
        )
    try:
        if not artifacts_path.is_dir() or not any(artifacts_path.iterdir()):
            raise ValueError("--artifacts-path должен быть существующим непустым каталогом")
    except OSError:
        raise ValueError("Не удалось прочитать каталог --artifacts-path") from None
    missing = [name for name in PDF_MODEL_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError(
            "PDF-конвейер Docling требует пакеты моделей (docling-slim[models-local]); не установлены: "
            + ", ".join(missing) + ". Установка требует отдельного разрешения"
        )


def prepare_output_directory(directory: Path, names: OutputNames) -> Path:
    if not directory.is_dir():
        raise ValueError("--output-dir должен быть существующим каталогом")
    existing = [name for name in names.all() if (directory / name).exists()]
    if existing:
        raise ValueError(
            "В --output-dir уже есть результаты этого ID: " + ", ".join(existing)
            + ". Перезапись не выполняется; удалите их вручную или выберите другой каталог/ID"
        )
    return directory


def write_new_file(directory: Path, name: str, payload: bytes) -> None:
    """Атомарно создать файл; существующий файл не перезаписывается."""
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, directory / name)
        finally:
            os.unlink(temporary)
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError:
        raise ValueError(f"Файл {name} появился во время работы; перезапись не выполняется") from None
    except OSError:
        raise ValueError(f"Не удалось атомарно записать {name} в --output-dir") from None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in VERSIONED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_converter(input_format: str, artifacts_path: Path | None) -> tuple[Any, dict[str, Any]]:
    """Собрать DocumentConverter только для ожидаемого формата и вернуть его настройки для отчёта."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import ConvertPipelineOptions, PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
    except ImportError:
        raise ValueError(
            "Docling не установлен; нужна необязательная зависимость ingest (см. README)"
        ) from None
    docling_format = InputFormat(input_format)
    if input_format == "pdf":
        pipeline_options: Any = PdfPipelineOptions(
            artifacts_path=artifacts_path, do_ocr=False,
            enable_remote_services=False, allow_external_plugins=False,
        )
        format_option: Any = PdfFormatOption(pipeline_options=pipeline_options)
    else:
        pipeline_options = ConvertPipelineOptions(enable_remote_services=False, allow_external_plugins=False)
        format_option = WordFormatOption(pipeline_options=pipeline_options)
    converter = DocumentConverter(
        allowed_formats=[docling_format], format_options={docling_format: format_option},
    )
    settings = {
        "pipeline": format_option.pipeline_cls.__name__,
        "backend": format_option.backend.__name__,
        "artifacts_path_provided": artifacts_path is not None,
        "pipeline_options": pipeline_options.model_dump(mode="json", exclude={"artifacts_path"}),
    }
    return converter, settings


def structure_counts(document: Any) -> dict[str, Any]:
    """Наблюдаемые счётчики элементов DoclingDocument без текста документа."""
    from docling_core.types.doc import ContentLayer, DocItemLabel

    items = [*document.texts, *document.tables, *document.pictures,
             *document.key_value_items, *document.form_items]
    reachable = sum(
        1 for _ in document.iterate_items(with_groups=False, included_content_layers=set(ContentLayer))
    )
    return {
        "items_by_label": dict(sorted(Counter(item.label.value for item in items).items())),
        "items_by_content_layer": dict(sorted(Counter(item.content_layer.value for item in items).items())),
        "groups_by_label": dict(sorted(Counter(group.label.value for group in document.groups).items())),
        "section_header_levels": dict(sorted(Counter(
            str(item.level) for item in document.texts if item.label == DocItemLabel.SECTION_HEADER
        ).items())),
        "items_total": len(items),
        "items_reachable_from_tree": reachable,
        "text_characters_total": sum(len(item.text) for item in document.texts),
        "table_cell_characters_total": sum(
            len(cell.text) for table in document.tables for cell in table.data.table_cells
        ),
        "pages": len(document.pages),
    }


def _conversion_errors(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "component": error.component_type.value, "module": error.module_name,
            "category": error.category.value, "page_no": error.page_no,
            "message": _bounded(error.error_message),
        }
        for error in result.errors[:MAX_DIAGNOSTICS]
    ]


def convert_source(
    source: SourceFile, converter: Any, max_bytes: int, max_pages: int,
) -> tuple[Any | None, dict[str, Any]]:
    """Выполнить преобразование с перехватом журнала/предупреждений; вернуть документ и диагностику."""
    from docling.datamodel.base_models import DocumentStream

    handler = _DiagnosticsHandler()
    previous_call_handlers = logging.Logger.callHandlers

    def capture_record(logger: logging.Logger, record: logging.LogRecord) -> None:
        if record.levelno >= handler.level:
            handler.handle(record)

    # Одно преобразование в отдельном CLI-процессе: временно заменяем маршрутизацию,
    # включая дочерние/новые Logger и propagate=False. Уровни, фильтры и списки
    # обработчиков не меняем.
    logging.Logger.callHandlers = capture_record
    started = time.monotonic()
    result = None
    exception: dict[str, str] | None = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = converter.convert(
                    DocumentStream(name=source.filename, stream=BytesIO(source.data)),
                    raises_on_error=False, max_num_pages=max_pages, max_file_size=max_bytes,
                )
            except Exception as error:  # Отказ Docling фиксируется в отчёте как FAILURE.
                exception = {"type": type(error).__name__, "message": _bounded(str(error))}
    finally:
        logging.Logger.callHandlers = previous_call_handlers
        handler.close()
    elapsed = time.monotonic() - started
    diagnostics: dict[str, Any] = {
        "elapsed_seconds": round(elapsed, 3),
        "docling_status": result.status.value if result is not None else None,
        "detected_format": result.input.format.value if result is not None and result.input.format else None,
        "exception": exception,
        "errors_total": len(result.errors) if result is not None else 0,
        "errors": _conversion_errors(result) if result is not None else [],
        "log_counts": dict(sorted(handler.counts.items())),
        "log_records": handler.records,
        "python_warnings_total": len(caught),
        "python_warnings": [
            {"category": warning.category.__name__, "message": _bounded(str(warning.message))}
            for warning in caught[:MAX_DIAGNOSTICS]
        ],
    }
    document = result.document if result is not None else None
    return document, diagnostics


def classify(source: SourceFile, document: Any | None, diagnostics: dict[str, Any], counts: dict[str, Any] | None) -> tuple[str, str]:
    """Итог: complete, incomplete или failed с причиной; полнота никогда не подразумевается."""
    status = diagnostics["docling_status"]
    if diagnostics["exception"] is not None or document is None:
        return "failed", "Docling завершился исключением или не вернул документ"
    if diagnostics["detected_format"] != source.input_format:
        return "failed", "Формат, определённый Docling, не совпадает с расширением"
    if status not in {"success", "partial_success"}:
        return "failed", f"Статус Docling: {status}"
    if counts is None or counts["text_characters_total"] + counts["table_cell_characters_total"] == 0:
        return "failed", "В снимке нет текстового содержимого"
    if status == "partial_success":
        return "incomplete", "Docling вернул PARTIAL_SUCCESS: снимок неполный, только для ручного разбора"
    return "complete", "Docling вернул SUCCESS; текстовая и структурная точность не проверены"


def _resource_usage() -> dict[str, float | int]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "cpu_seconds_self": round(own.ru_utime + own.ru_stime, 3),
        "cpu_seconds_children": round(children.ru_utime + children.ru_stime, 3),
        "peak_rss_kib_self": own.ru_maxrss,
        "peak_rss_kib_largest_child": children.ru_maxrss,
    }


def _print_summary(document_id: str, source: SourceFile, outcome: str, reason: str,
                   diagnostics: dict[str, Any], counts: dict[str, Any] | None,
                   written: list[str], snapshot_sha256: str | None) -> None:
    print("СТРУКТУРНЫЙ СНИМОК DOCLING — производное представление, не юридический оригинал; точность не проверена")
    print(f"Документ: {document_id}; формат: {source.input_format.upper()}; размер: {len(source.data)} байт")
    print(f"SHA-256 исходных байтов: {source.sha256}")
    print(f"Статус Docling: {diagnostics['docling_status']}; итог: {outcome.upper()} — {reason}")
    if counts is not None:
        labels = counts["items_by_label"]
        print(
            f"Элементы: всего {counts['items_total']}, в дереве {counts['items_reachable_from_tree']}; "
            f"заголовков {labels.get('section_header', 0) + labels.get('title', 0)}, "
            f"элементов списков {labels.get('list_item', 0)}, таблиц {labels.get('table', 0)}, "
            f"сносок {labels.get('footnote', 0)}; символов текста {counts['text_characters_total']}"
        )
    categories = Counter(error["category"] for error in diagnostics["errors"])
    print(
        f"Диагностика: ошибок Docling {diagnostics['errors_total']} {dict(categories) or ''}; "
        f"записей журнала {sum(diagnostics['log_counts'].values())}; "
        f"предупреждений Python {diagnostics['python_warnings_total']}"
        + (f"; исключение {diagnostics['exception']['type']}" if diagnostics["exception"] else "")
    )
    if snapshot_sha256 is not None:
        print(f"SHA-256 сериализованного снимка: {snapshot_sha256}")
    print(f"Время преобразования: {diagnostics['elapsed_seconds']} с; записаны: {', '.join(written)}")


class _RussianArgumentParser(argparse.ArgumentParser):
    """Локализованные ошибки CLI без вывода значений аргументов."""

    def format_usage(self) -> str:
        return super().format_usage().replace("usage: ", "Использование: ", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage: ", "Использование: ", 1)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, "Ошибка аргументов: проверьте имена и значения параметров. "
                  "Обязательны input, --document-id и --output-dir. Справка: --help.\n")


def main(argv: list[str] | None = None) -> int:
    parser = _RussianArgumentParser(
        prog="python -m llm_wiki.ingest", add_help=False,
        description="Локальное структурное преобразование одного RTF/DOCX/PDF через Docling в JSON-снимок "
        "DoclingDocument и отдельный отчёт. Без LLM и сети; запуск только в изолированном сетевом "
        "пространстве (например, unshare -rn).",
    )
    parser._positionals.title = "Позиционные аргументы"
    parser._optionals.title = "Параметры"
    parser.add_argument("-h", "--help", action="help", help="показать справку и выйти")
    parser.add_argument("input", type=Path, help="путь к локальному .rtf, .docx или .pdf (регистр расширения не важен)")
    parser.add_argument("--document-id", required=True, help="доверенный ID источника (ASCII-буквы/цифры/_/-)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="существующий игнорируемый Git каталог для снимка и отчёта, например runtime/ingestion")
    parser.add_argument("--max-bytes", default=str(DEFAULT_MAX_BYTES),
                        help=f"предел размера источника в байтах, 1–{MAX_BYTES_LIMIT}; по умолчанию {DEFAULT_MAX_BYTES}")
    parser.add_argument("--max-pages", default=str(DEFAULT_MAX_PAGES),
                        help=f"предел страниц PDF, 1–{MAX_PAGES_LIMIT}; по умолчанию {DEFAULT_MAX_PAGES}; для RTF/DOCX не применяется")
    parser.add_argument("--artifacts-path", type=Path,
                        help="только PDF: локальный каталог заранее полученных моделей Docling; обязателен для PDF")
    parser.add_argument("--libreoffice", type=Path,
                        help="только RTF: путь к soffice; по умолчанию поиск soffice в PATH")
    args = parser.parse_args(argv)
    try:
        validate_document_id(args.document_id)
        max_bytes = _bounded_int(args.max_bytes, "--max-bytes", MAX_BYTES_LIMIT)
        max_pages = _bounded_int(args.max_pages, "--max-pages", MAX_PAGES_LIMIT)
        names = OutputNames.for_document(args.document_id)
        output_dir = prepare_output_directory(args.output_dir, names)
        source = read_source_file(args.input, max_bytes)
        require_network_isolation()
        libreoffice_version = None
        if source.input_format == "rtf":
            libreoffice_cmd, libreoffice_version = resolve_libreoffice(args.libreoffice)
            os.environ["DOCLING_LIBREOFFICE_CMD"] = libreoffice_cmd
        if source.input_format == "pdf":
            check_pdf_prerequisites(args.artifacts_path)
        os.environ.update(OFFLINE_ENVIRONMENT)
        converter, settings = build_converter(source.input_format, args.artifacts_path)
        document, diagnostics = convert_source(source, converter, max_bytes, max_pages)
        counts = structure_counts(document) if document is not None else None
        outcome, reason = classify(source, document, diagnostics, counts)

        written: list[str] = []
        snapshot: dict[str, Any] | None = None
        if outcome != "failed":
            payload = json.dumps(document.export_to_dict(), ensure_ascii=False).encode("utf-8")
            snapshot_name = names.snapshot if outcome == "complete" else names.incomplete_snapshot
            write_new_file(output_dir, snapshot_name, payload)
            written.append(snapshot_name)
            snapshot = {
                "file": snapshot_name, "complete": outcome == "complete",
                "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
                "serialization": "DoclingDocument.export_to_dict() → JSON UTF-8 без форматирования",
            }
        report = {
            "report_schema": REPORT_SCHEMA,
            "document_id": args.document_id,
            "outcome": outcome, "reason": reason,
            "manual_review_required": True, "fidelity": "NOT_VERIFIED",
            "source": {
                "filename": source.filename, "format_by_extension": source.input_format,
                "size_bytes": len(source.data), "sha256": source.sha256,
                "hash_scope": "исходные байты файла до преобразования",
            },
            "intermediate": (
                "RTF преобразован LibreOffice в DOCX внутри Docling в памяти; промежуточный DOCX не сохраняется и не хешируется"
                if source.input_format == "rtf" else None
            ),
            "snapshot": snapshot,
            "structure": counts,
            "conversion": {
                **settings,
                "limits": {
                    "max_file_size_bytes": max_bytes, "max_num_pages": max_pages,
                    "max_num_pages_applies": source.input_format == "pdf",
                    "hard_limits": "только тайм-аут subprocess LibreOffice внутри Docling для RTF; "
                    "время, CPU и RAM процесса Python не ограничены — используйте внешнюю границу процесса",
                },
                "offline": {
                    "enable_remote_services": False, "allow_external_plugins": False,
                    "environment": OFFLINE_ENVIRONMENT,
                    "network_check": "до преобразования видны только loopback-интерфейсы",
                    "note": "Флаги Docling сами по себе не запрещают загрузку весов моделей",
                },
            },
            "diagnostics": diagnostics,
            "runtime": {
                "python": platform.python_version(), "system": platform.system(),
                "machine": platform.machine(), "packages": _package_versions(),
                "libreoffice": libreoffice_version,
            },
            "resources_observed": _resource_usage(),
            "provenance_limits": list(PROVENANCE_LIMITS),
        }
        report_payload = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        write_new_file(output_dir, names.report, report_payload)
        written.append(names.report)
        _print_summary(args.document_id, source, outcome, reason, diagnostics, counts, written,
                       snapshot["sha256"] if snapshot else None)
        return 0 if outcome == "complete" else 1
    except ValueError as error:
        print(f"Ошибка: {error}. Обработка остановлена; перезапись и публикация не выполнялись.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
