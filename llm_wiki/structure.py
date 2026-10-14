"""Предварительная правовая структура принятого снимка Docling: этапы 1.2a и 1.2a.1-A.

Без разбора оригинала, ссылок, LLM и юридической верификации. Ссылки и порядок
относятся к снимку; кандидаты не задают границ полного нормативного положения.
Границы и домены нумерации (1.2a.1-A) строятся в `boundaries`; иерархия по позиции
и присоединение продолжений не выполняются.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Literal
import warnings

from . import boundaries
from .boundaries import CHAPTER, CONTENT_LABELS, StreamItem, article_match
from .validation import validate_document_id

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

SCHEMA = "llm-wiki-memory/legal-structure/v2"
REPORT_SCHEMA = "llm-wiki-memory/structure-report/v2"
# Политика включает буквы инкремента: 1.2a.1-B изменит правила и обязан сменить версию.
POLICY = "legal-cues/1.2a.1-A-v2"
CORE_VERSION = "2.101.1"
DOCLING_SCHEMA_VERSION = "1.10.0"
NODE_LISTS = ("groups", "texts", "pictures", "tables", "key_value_items", "form_items",
              "field_regions", "field_items")
REPOSITORY = Path(__file__).resolve().parent.parent


class StructureError(ValueError):
    """Контролируемая ошибка без исходного текста и частных путей."""


@dataclass(frozen=True)
class SnapshotIdentity:
    document_id: str
    source_sha256: str
    snapshot_sha256: str
    report_sha256: str
    docling_schema_version: str
    docling_core_version: str
    policy: str = POLICY
    fidelity: str = "NOT_VERIFIED"


# Машинные коды единиц и учёта блоков; коды границ — в boundaries.CODES.
CODES: dict[str, str] = {
    "snapshot_container": "контейнер снимка, не установленная юридическая идентичность",
    "isolated_chapter_marker": "изолированное «ГЛАВА N»",
    "article_marker_with_delimiter": "явное «Статья N» с разделителем",
    "root": "корень",
    "docling_ancestry": "явная ancestry Docling; предварительная, не юридически проверенная связь",
    "unknown_intermediate_node": "неизвестный промежуточный узел; родитель не определён",
    "unsupported_kind_pair": "родитель неизвестен или сочетание типов не поддерживается политикой",
    "unit_extent_unknown": "полная граница единицы не установлена",
    "parent_unknown": "родитель не определён",
    "duplicate_designation_in_parent": "повтор обозначения в одном родительском контексте",
    "role_unknown": "тип или назначение не установлены",
    "empty_text": "пустой текстовый элемент; это не классификация соседнего содержания",
    "table_or_picture_content": "содержимое таблицы/изображения; правовая роль не установлена",
    "furniture_or_other_layer": "furniture или иной слой; правовая роль не установлена",
    "complex_inline_group": "сложный InlineGroup; требуется рассмотрение",
    "other_label_content": "сноска, подпись, код или иная метка вне правил единиц",
    "no_supported_marker": "нет поддержанного однозначного правового маркера",
    "boundary_marker_only": "маркер границы без правовой единицы",
    "numbering_marker_only": "маркер нумерации без правовой единицы и без родителя",
    "unit_anchor": "якорь предварительной единицы; продолжение не присоединяется",
    "rich_cell_ref": "rich-cell ref; текст повторно не учитывается",
    "table_cell_unknown": "ячейка таблицы; назначение не установлено",
    "empty_cell": "пустая ячейка",
}


@dataclass(frozen=True)
class LegalUnit:
    id: str
    kind: Literal["document", "chapter", "article"]
    number: str | None
    source_refs: tuple[str, ...]
    parent_id: str | None
    recognition: str
    parent_basis: str
    uncertainty: tuple[str, ...] = ()


@dataclass
class BlockAccounting:
    source_ref: str
    label: str
    status: Literal["assigned", "non_normative", "unresolved"]
    reason: str
    unit_id: str | None = None
    represented_by: str | None = None


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructureError("Повторный ключ JSON")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise StructureError("Недопустимое число JSON")


def _read_json(path: Path, limit: int) -> tuple[bytes, dict[str, Any]]:
    try:
        if not path.is_file():
            raise StructureError("Вход должен быть обычным существующим файлом")
        with path.open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise StructureError("Вход должен быть обычным файлом")
            data = stream.read(limit + 1)
        if not data or len(data) > limit:
            raise StructureError("Пустой вход или превышен предел размера JSON")
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise StructureError("Ожидается объект JSON")
        return data, value
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise StructureError("Не удалось прочитать корректный UTF-8 JSON") from None


def _inventory(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Проверить адреса, все ссылки и дерево до рекурсивной загрузки Docling."""
    nodes = {"#/body": raw["body"], "#/furniture": raw["furniture"]}
    for name in NODE_LISTS:
        values = raw.get(name, [])
        if not isinstance(values, list):
            raise StructureError("Неверный список узлов снимка")
        nodes.update((f"#/{name}/{i}", item) for i, item in enumerate(values))
    for ref, node in nodes.items():
        if not isinstance(node, dict) or node.get("self_ref") != ref:
            raise StructureError("Адрес узла не совпадает с его положением в снимке")
    stack: list[Any] = [raw]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if "$ref" in value and (set(value) != {"$ref"} or value["$ref"] not in nodes):
                raise StructureError("Отсутствующая или неподдерживаемая ссылка снимка")
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    visited: set[str] = set()
    pending = [("#/furniture", None, 0), ("#/body", None, 0)]
    while pending:
        ref, parent, depth = pending.pop()
        if ref in visited or depth > 100:
            raise StructureError("Повтор узла, цикл или глубина дерева больше 100")
        visited.add(ref)
        node = nodes[ref]
        if node.get("parent") != ({"$ref": parent} if parent else None):
            raise StructureError("Родитель и children снимка не согласованы")
        children = node.get("children", [])
        if not isinstance(children, list):
            raise StructureError("Неверный список children")
        for child in reversed(children):
            if not isinstance(child, dict) or set(child) != {"$ref"}:
                raise StructureError("Неверная ссылка children")
            pending.append((child["$ref"], ref, depth + 1))
    if visited != nodes.keys():
        raise StructureError("В снимке есть узлы вне body/furniture; принятие без учёта запрещено")
    # Rich-cell ref — ссылка на уже учтённого потомка таблицы, не второй текст.
    rich_targets: set[str] = set()
    for table in raw.get("tables", []):
        for cell in table["data"]["table_cells"]:
            if cell.get("ref"):
                target = cell["ref"]["$ref"]
                if target in rich_targets:
                    raise StructureError("Неоднозначный повтор rich-cell ref")
                rich_targets.add(target)
                ancestor = nodes[target].get("parent")
                while ancestor and ancestor["$ref"] != table["self_ref"]:
                    ancestor = nodes[ancestor["$ref"]].get("parent")
                if not ancestor:
                    raise StructureError("Rich-cell ref не принадлежит своей таблице")
    return nodes


def load_snapshot(snapshot: Path, report_path: Path) -> tuple["DoclingDocument", SnapshotIdentity, int]:
    """Принять согласованную пару; хеш не аутентифицирует автора отчёта или оригинал."""
    data, raw = _read_json(snapshot, 100 * 1024 * 1024)
    report_data, report = _read_json(report_path, 2 * 1024 * 1024)
    try:
        if (report["report_schema"] != "llm-wiki-memory/ingest-report/v1"
                or report["outcome"] != "complete" or report["snapshot"]["complete"] is not True
                or report["fidelity"] != "NOT_VERIFIED"
                or report["manual_review_required"] is not True):
            raise StructureError("Не поддерживается отчёт или снимок не завершён")
        document_id = report["document_id"]
        validate_document_id(document_id)
        details = report["snapshot"]
        if (snapshot.name != f"{document_id}.docling.json" or details["file"] != snapshot.name
                or report_path.name != f"{document_id}.report.json"
                or details["sha256"] != _digest(data)
                or type(details["size_bytes"]) is not int or details["size_bytes"] != len(data)):
            raise StructureError("Имя, размер или SHA-256 снимка не совпадает с отчётом")
        if not re.fullmatch(r"[0-9a-f]{64}", report["source"]["sha256"]):
            raise StructureError("Неверный SHA-256 источника в отчёте")
        if (raw["schema_name"] != "DoclingDocument" or raw["version"] != DOCLING_SCHEMA_VERSION):
            raise StructureError("Версия схемы Docling не поддерживается политикой 1.2a")
        source = report["source"]
        if (type(source["size_bytes"]) is not int or source["size_bytes"] <= 0
                or source["format_by_extension"] not in {"rtf", "docx", "pdf"}
                or not isinstance(source["filename"], str) or not source["filename"]
                or Path(source["filename"]).name != source["filename"]
                or report["diagnostics"]["docling_status"] != "success"
                or report["diagnostics"]["exception"] is not None):
            raise StructureError("Отчёт содержит неподдерживаемое или противоречивое происхождение")
        nodes = _inventory(raw)
        from docling_core.types.doc import DoclingDocument
        from pydantic import ValidationError

        core_version = importlib.metadata.version("docling-core")
        if core_version != CORE_VERSION:
            raise StructureError("Версия docling-core не проверена для этой политики; см. README")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                document = DoclingDocument.model_validate(raw)
        except (ValidationError, RuntimeError, RecursionError):
            raise StructureError("Схема или дерево Docling не прошли проверку; содержимое не выводится") from None
        # Загрузчик может мигрировать схемы: запрещаем изменение значимых полей.
        exported = document.export_to_dict()
        loaded = _inventory(exported)
        if nodes.keys() != loaded.keys() or any(
            any(node.get(key) != loaded[ref].get(key)
                for key in ("text", "label", "parent", "children", "data", "content_layer"))
            for ref, node in nodes.items()
        ):
            raise StructureError("Загрузчик изменил значимые поля; нужен отдельный разбор совместимости")
        if not (any(item.text.strip() for item in document.texts)
                or any(cell.text.strip() for table in document.tables for cell in table.data.table_cells)):
            raise StructureError("Снимок не содержит непустого текста")
        identity = SnapshotIdentity(document_id, report["source"]["sha256"], _digest(data),
                                    _digest(report_data), raw["version"], core_version)
        return document, identity, len(caught)
    except (KeyError, TypeError, AttributeError, IndexError):
        raise StructureError("Некорректная структура снимка или отчёта; содержимое не выводится") from None
    except ImportError:
        raise StructureError("Нужен установленный docling-core из extra ingest; установка не выполняется") from None


def recognize(document: "DoclingDocument", identity: SnapshotIdentity) -> dict[str, Any]:
    """Единицы 1.2a, границы и домены 1.2a.1-A; порядок — body, затем отдельно furniture."""
    from docling_core.types.doc import ContentLayer, DocItem, GroupItem, InlineGroup, TextItem

    ordered = [item for root in (document.body, document.furniture)
               for item, _ in document.iterate_items(
                   root=root, with_groups=True, traverse_pictures=True,
                   included_content_layers=set(ContentLayer))]
    nodes = {item.self_ref: item for item in ordered}
    if len(nodes) != len(ordered):
        raise StructureError("Обход Docling повторяет узлы")
    scope = _digest(f"{identity.snapshot_sha256}:{identity.policy}:{identity.docling_core_version}".encode())

    def unit_id(ref: str) -> str:
        return "unit-" + _digest(f"{scope}:{ref}".encode())

    def make_id(prefix: str, *parts: str) -> str:
        return f"{prefix}-" + _digest(":".join((scope, prefix, *parts)).encode())

    def ancestors(item: Any) -> list[Any]:
        result = []
        while item.parent:
            item = nodes[item.parent.cref]
            result.append(item)
        return result

    root_id = unit_id("#/body")
    units = [LegalUnit(root_id, "document", None, ("#/body",), None, "snapshot_container", "root")]
    unit_by_ref = {"#/body": root_id}
    blocks: dict[str, BlockAccounting] = {}
    for item in ordered:
        ref = item.self_ref
        blocks[ref] = BlockAccounting(ref, item.label.value, "unresolved", "role_unknown")
        if isinstance(item, TextItem) and not item.text.strip():
            blocks[ref].status = "non_normative"
            blocks[ref].reason = "empty_text"
    blocks["#/body"] = BlockAccounting("#/body", document.body.label.value, "assigned",
                                       "snapshot_container", root_id)
    # Поток body в порядке обхода: один элемент на логический текст; таблица/изображение —
    # один непрозрачный элемент без потомков; furniture в поток не входит.
    stream: list[StreamItem] = []
    consumed: set[str] = set()
    for order, item in enumerate(ordered):
        ref = item.self_ref
        chain = ancestors(item)
        if any(ancestor.self_ref.startswith(("#/tables/", "#/pictures/")) for ancestor in chain):
            if blocks[ref].status == "unresolved":
                blocks[ref].reason = "table_or_picture_content"
            continue
        if ref in consumed:
            continue
        in_furniture = any(a.self_ref == "#/furniture" for a in chain) or ref == "#/furniture"
        if in_furniture or item.content_layer.value != "body":
            if blocks[ref].status == "unresolved":
                blocks[ref].reason = "furniture_or_other_layer"
            if not in_furniture and not isinstance(item, GroupItem):
                stream.append(StreamItem(order, (ref,), "other_layer", item.label.value))
            continue
        refs = [ref]
        if isinstance(item, InlineGroup):
            children = [nodes[r.cref] for r in item.children]
            if not children or any(not isinstance(c, TextItem) or c.children for c in children):
                blocks[ref].reason = "complex_inline_group"
                consumed.update(c.self_ref for c in children)
                stream.append(StreamItem(order, (ref,), "unsupported", item.label.value))
                continue
            refs.extend(c.self_ref for c in children)
            consumed.update(refs[1:])
            text = " ".join(c.text for c in children)
            for child in children:
                blocks[child.self_ref].represented_by = ref
        elif isinstance(item, TextItem):
            if any(isinstance(a, InlineGroup) for a in chain):
                continue
            text = item.text
        else:
            if isinstance(item, DocItem):
                stream.append(StreamItem(order, (ref,), "opaque", item.label.value))
            continue
        # Только представление для распознавания: исходные строки не меняются/не копируются.
        cue = re.sub(r"\s+", " ", text).strip()
        if not cue:
            continue
        if item.label.value not in CONTENT_LABELS:
            stream.append(StreamItem(order, tuple(refs), "other_label", item.label.value))
            for source_ref in refs:
                if blocks[source_ref].status == "unresolved":
                    blocks[source_ref].reason = "other_label_content"
            continue
        stream.append(StreamItem(order, tuple(refs), "content", item.label.value, cue,
                                 bool(getattr(item, "enumerated", False))))
        chapter = CHAPTER.fullmatch(cue)
        article = article_match(cue)
        match = chapter or article
        if not match:
            for source_ref in refs:
                if blocks[source_ref].status == "unresolved":
                    blocks[source_ref].reason = "no_supported_marker"
            continue
        kind = "chapter" if chapter else "article"
        # Не пропускать нераспознанный heading/текст при поиске родителя.
        parent_id = None
        parent_basis = "unknown_intermediate_node"
        for ancestor in chain:
            if ancestor.self_ref in unit_by_ref:
                parent_id = unit_by_ref[ancestor.self_ref]
                parent_basis = "docling_ancestry"
                break
            if not isinstance(ancestor, GroupItem):
                break
        parent_kind = next((u.kind for u in units if u.id == parent_id), None)
        if parent_kind not in ({"document"} if kind == "chapter" else {"document", "chapter"}):
            parent_id = None
            parent_basis = "unsupported_kind_pair"
        current_id = unit_id(ref)
        units.append(LegalUnit(current_id, kind, match.group(1), tuple(refs), parent_id,
                               "isolated_chapter_marker" if chapter else "article_marker_with_delimiter",
                               parent_basis, ("unit_extent_unknown",)
                               + (("parent_unknown",) if parent_id is None else ())))
        for source_ref in refs:
            unit_by_ref[source_ref] = current_id
            blocks[source_ref].status = "assigned"
            blocks[source_ref].unit_id = current_id
            blocks[source_ref].reason = "unit_anchor"

    found = boundaries.analyze(stream, make_id)
    # Маркер внутри предполагаемой цитаты не подавляется, но единица получает неопределённость.
    quoted_by_ref = {b.source_refs[0]: b.quoted_context for b in found["boundaries"]
                     if b.category in {"chapter", "article"} and b.quoted_context != "none"}
    repeated = Counter((u.parent_id, u.kind, u.number) for u in units if u.number is not None)
    units = [replace(u, uncertainty=(
                 *u.uncertainty,
                 *(("duplicate_designation_in_parent",) if u.number is not None
                   and repeated[(u.parent_id, u.kind, u.number)] > 1 else ()),
                 *((f"quoted_context_{quoted_by_ref[u.source_refs[0]]}",)
                   if u.source_refs[0] in quoted_by_ref else ())))
             for u in units]
    for kind, records in (("boundary_marker_only", found["boundaries"]), ("numbering_marker_only",
                                                                         found["numbering_markers"])):
        for record in records:
            for source_ref in record.source_refs:
                if blocks[source_ref].reason == "no_supported_marker":
                    blocks[source_ref].reason = kind
    # Ячейки имеют свои адреса; rich ref обозначает альтернативное представление,
    # а не добавочный абзац. Порядок ячеек — порядок сериализации, не правовой порядок.
    cells = []
    for table in document.tables:
        for index, cell in enumerate(table.data.table_cells):
            rich_ref = getattr(cell, "ref", None)
            cells.append(BlockAccounting(
                f"{table.self_ref}/data/table_cells/{index}", "table_cell",
                "unresolved" if cell.text.strip() or rich_ref else "non_normative",
                "rich_cell_ref" if rich_ref else "table_cell_unknown" if cell.text.strip() else "empty_cell",
                represented_by=rich_ref.cref if rich_ref else None,
            ))
    result = {
        "schema": SCHEMA, "identity": asdict(identity), "analysis_id": scope,
        "status": "PRELIMINARY", "manual_review_required": True,
        "units": [asdict(unit) for unit in units],
        "blocks": [asdict(block) for block in blocks.values()],
        "table_cells": [asdict(cell) for cell in cells],
        "boundaries": [asdict(boundary) for boundary in found["boundaries"]],
        "numbering_markers": [asdict(marker) for marker in found["numbering_markers"]],
        "quoted_spans": [asdict(span) for span in found["quoted_spans"]],
        "numbering_domains": [asdict(domain) for domain in found["numbering_domains"]],
        "observations": {"title_after_heading_not_boundary": found["stats"]["after_heading_not_boundary"]},
        "codes": dict(sorted({**CODES, **boundaries.CODES}.items())),
    }
    _check_invariants(result, [item.order for item in stream])
    return result


def _check_invariants(result: dict[str, Any], stream_orders: list[int]) -> None:
    """Внутренние инварианты v2: при нарушении результат не принимается (fail closed)."""
    codes = result["codes"]
    used = [u[key] for u in result["units"] for key in ("recognition", "parent_basis")]
    used += [code for u in result["units"] for code in u["uncertainty"]]
    used += [b["reason"] for b in (*result["blocks"], *result["table_cells"])]
    used += [b[key] for b in result["boundaries"]
             for key in ("category", "effect", "evidence", "confidence", "quoted_context")]
    used += [code for b in result["boundaries"] for code in b["uncertainty"]]
    used += [m[key] for m in result["numbering_markers"]
             for key in ("form", "sequence", "verification", "quoted_context")]
    used += [s["end_basis"] for s in result["quoted_spans"]]
    used += [d[key] for d in result["numbering_domains"]
             for key in ("start_basis", "end_basis", "confidence", "quoted_context", "continuity")]
    used += [code for d in result["numbering_domains"] for code in d["sequence_issues"]]
    if any(code not in codes for code in used):
        raise StructureError("Внутренний код вне словаря codes; результат не принят")
    ids = [record["id"] for key in ("units", "boundaries", "numbering_markers", "quoted_spans",
                                    "numbering_domains") for record in result[key]]
    refs = {block["source_ref"] for block in result["blocks"]}
    records = [*result["boundaries"], *result["numbering_markers"]]
    orders = [(b["order"], b["category"]) for b in result["boundaries"]]
    domains = result["numbering_domains"]
    covered = [order for d in domains for order in stream_orders if d["start_order"] <= order <= d["end_order"]]
    by_id = {d["id"]: d for d in domains}
    if (len(ids) != len(set(ids))
            or any(ref not in refs for record in records for ref in record["source_refs"])
            or orders != sorted(orders)
            or covered != stream_orders
            or any(a["end_order"] >= b["start_order"] or a["closed_by"] != b["opened_by"]
                   for a, b in zip(domains, domains[1:]))
            or any(not by_id[m["domain_id"]]["start_order"] <= m["order"] <= by_id[m["domain_id"]]["end_order"]
                   for m in result["numbering_markers"])
            or any(b["effect"] == "domain_boundary" and b["id"] not in {d["opened_by"] for d in domains}
                   for b in result["boundaries"])
            or any(block["status"] == "assigned" and block["reason"] not in {"unit_anchor", "snapshot_container"}
                   for block in result["blocks"])):
        raise StructureError("Нарушен инвариант границ, доменов или учёта блоков; результат не принят")


def _output_directory(path: Path, names: tuple[str, str]) -> Path:
    """Новый CLI пишет только в существующий частный игнорируемый каталог runtime."""
    directory = path.resolve()
    if (not directory.is_relative_to(REPOSITORY / "runtime") or not directory.is_dir()
            or directory.stat().st_mode & 0o077):
        raise StructureError("Нужен существующий каталог внутри runtime с правами 0700")
    for name in names:
        target = directory / name
        if target.exists() or target.is_symlink():
            raise StructureError("Результат уже существует; выберите другой частный каталог")
        result = subprocess.run(["git", "check-ignore", "-q", "--", str(target)],
                                cwd=REPOSITORY, capture_output=True)
        if result.returncode != 0:
            raise StructureError("Результат должен игнорироваться Git и не быть отслеживаемым")
    return directory


def _write_new(directory: Path, name: str, payload: bytes) -> None:
    """Атомарная запись одного файла 0600 без перезаписи, как в ingestion."""
    descriptor, temporary = tempfile.mkstemp(prefix=".structure-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, directory / name)
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        os.unlink(temporary)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


class _Parser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        return super().format_usage().replace("usage: ", "Использование: ", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage: ", "Использование: ", 1)

    def error(self, message: str) -> None:
        self.exit(2, "Ошибка параметров; справка: python -m llm_wiki.structure --help\n")


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(prog="python -m llm_wiki.structure", add_help=False,
                     description="Предварительная структура принятого Docling JSON: единицы 1.2a, "
                     "границы и домены нумерации 1.2a.1-A (контракт legal-structure/v2). Без иерархии "
                     "по позиции, оригинала, LLM и юридической проверки; текст в терминал не выводится.")
    parser._positionals.title = "Позиционные аргументы"
    parser._optionals.title = "Параметры"
    parser.add_argument("-h", "--help", action="help", help="показать справку")
    parser.add_argument("snapshot", type=Path, help="принятый локальный .docling.json")
    parser.add_argument("--report", required=True, type=Path, help="соответствующий ingestion .report.json")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="существующий игнорируемый каталог внутри runtime, права 0700")
    args = parser.parse_args(argv)
    try:
        document, identity, warning_count = load_snapshot(args.snapshot, args.report)
        with warnings.catch_warnings(record=True) as analysis_warnings:
            warnings.simplefilter("always")
            try:
                structure = recognize(document, identity)
            except (KeyError, IndexError, TypeError, AttributeError, StopIteration):
                raise StructureError("Неожиданная структура при анализе; содержимое не выводится") from None
        basename = structure["analysis_id"]
        names = (f"{basename}.structure.json", f"{basename}.report.json")
        directory = _output_directory(args.output_dir, names)
        payload = _json_bytes(structure)
        def count(key: str, *fields: str) -> dict[str, int]:
            return dict(sorted(Counter("/".join(str(record[field]) for field in fields)
                                       for record in structure[key]).items()))

        report = {
            "schema": REPORT_SCHEMA, "status": "PRELIMINARY",
            "identity": asdict(identity), "analysis_id": basename,
            "structure_file": names[0], "structure_sha256": _digest(payload),
            "counts": {
                "units": count("units", "kind"),
                "nodes": count("blocks", "status"),
                "table_cells": count("table_cells", "status"),
                "boundaries": count("boundaries", "category", "effect", "confidence"),
                "numbering_markers": count("numbering_markers", "form", "sequence"),
                "numbering_domains": count("numbering_domains", "start_basis", "confidence", "quoted_context"),
                "domain_continuity": count("numbering_domains", "continuity"),
                "quoted_spans": count("quoted_spans", "end_basis"),
                **structure["observations"],
            },
            "loader_warnings": warning_count,
            "analysis_warnings": len(analysis_warnings),
            "limits": ["Кандидаты и ancestry предварительны; нормативная полнота не установлена.",
                       "Границы и домены — интервалы потока снимка, не правовые единицы и не их содержание.",
                       "Нет иерархии по позиции, присоединения абзацев, разбора пунктов, ссылок и определений.",
                       "Номера вычисленного ListItem.marker не проверены и не используются.",
                       "Подписи/реквизиты не распознаются; цитаты определяются эвристически.",
                       "Порядок body и отдельно furniture — порядок снимка, не проверка оригинала.",
                       "Исходные outlinelevel/закладки не восстанавливаются; fidelity NOT_VERIFIED.",
                       "Хеши проверяют согласованность входов, не подлинность источника/отчёта."],
        }
        _write_new(directory, names[0], payload)
        # Отчёт пишется последним: без него пара артефактов не считается завершённой.
        _write_new(directory, names[1], _json_bytes(report))
        print("Предварительная структура записана; требуется ручное рассмотрение.")
        print(f"Кандидатов: {len(structure['units'])}; узлов: {len(structure['blocks'])}; "
              f"неразрешённых узлов: {report['counts']['nodes'].get('unresolved', 0)}; "
              f"границ: {len(structure['boundaries'])}; доменов нумерации: "
              f"{len(structure['numbering_domains'])}.")
        print(f"Файлы: {names[0]}, {names[1]}")
        return 0
    except StructureError as error:
        print(f"Ошибка: {error}. Структура не принята.", file=sys.stderr)
        return 1
    except (ValueError, OSError, RecursionError):
        # Исключения библиотек могут содержать исходный текст/пути: не печатаем их.
        print("Ошибка: вход, происхождение, дерево или каталог результатов не прошли проверку. "
              "Проверьте пару JSON, версии, права 0700 и отсутствие прежних результатов. "
              "Юридическая структура не принята.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
