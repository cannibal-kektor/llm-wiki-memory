"""LE-3A: точные доказательства (Evidence) и явный ContextPackage поверх снимка LE-2 (без ввода-вывода).

Доказательство — диапазон [start, end) символов Unicode проверенного фрагмента внутри одной строки
карты LE-2. Текст не хранится как каноническое состояние: он каждый раз заново вырезается из
фрагмента, а сериализация несёт только смещения и SHA-256 текста. Идентификатор доказательства
зависит от снимка и границ, но не от роли, утверждения или толкования модели.

ContextPackage — явно выбранный набор доказательств с ролями. Выбор нескольких строк не доказывает,
что учтены все применимые оговорки документа: полнота контекста всегда NOT_ESTABLISHED.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .source import Line, line_units

EVIDENCE_CONTRACT = "llm-wiki-memory/le3a-evidence/v1"
CONTEXT_CONTRACT = "llm-wiki-memory/le3a-context/v1"
SELECTION_CONTRACT = "llm-wiki-memory/le3a-selection/v1"
ROLES = ("TARGET", "SHARED_CONTEXT", "POTENTIAL_LIMITATION", "POTENTIAL_EXCEPTION", "UNRESOLVED_REFERENCE")
DEPENDENCY_KINDS = ("UNRESOLVED_REFERENCE", "OUT_OF_EXCERPT", "NOT_SEARCHED")
# Код примечания: только латиница, цифры и '_'; примечания попадают в пакет и в промпт.
NOTE_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
LINE_ID = re.compile(r"\d{1,3}:L\d{1,3}")
MAX_PACKAGE_EVIDENCE = 12
MAX_PACKAGE_CHARS = 8_000
MAX_DEPENDENCIES = 12
CONTEXT_COMPLETENESS = "NOT_ESTABLISHED"
# Единственный источник поля limits пакета: при загрузке требуется точное совпадение ключей, типов и значений.
PACKAGE_LIMITS = {"max_evidence": MAX_PACKAGE_EVIDENCE, "max_chars": MAX_PACKAGE_CHARS,
                  "max_dependencies": MAX_DEPENDENCIES, "multi_line_spans": "NOT_SUPPORTED",
                  "search_beyond_excerpt": "NOT_PERFORMED"}
EVIDENCE_FIELDS = frozenset(("evidence_id", "contract", "snapshot_id", "source_id", "excerpt_sha256", "line_id",
                             "point", "start", "end", "whole_line", "full_txt_start", "full_txt_end",
                             "text_sha256", "chars"))
PACKAGE_FIELDS = frozenset(("contract", "context_package_id", "snapshot", "target", "evidence",
                            "unresolved_dependencies", "derived_links", "context_completeness",
                            "limits", "selection_sha256", "legal_status", "review_status"))


class EvidenceError(Exception):
    """Доказательство или пакет не проходят проверку; сообщение не содержит текста источника."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    """Проверенный фрагмент LE-2 и его карта строк; snapshot_id выводится из значимых полей карты."""

    snapshot_id: str
    source_id: str
    txt_sha256: str
    excerpt_sha256: str
    excerpt_start: int
    excerpt: str
    units: tuple[Line, ...]

    def identity(self) -> dict[str, Any]:
        return {"snapshot_id": self.snapshot_id, "source_id": self.source_id, "txt_sha256": self.txt_sha256,
                "excerpt_sha256": self.excerpt_sha256,
                "excerpt_full_txt_range": [self.excerpt_start, self.excerpt_start + len(self.excerpt)]}


def snapshot(excerpt: str, source_map: dict[str, Any]) -> Snapshot:
    """Собрать снимок; карта строк заново выводится из фрагмента и должна совпасть с записанной."""
    try:
        recorded = source_map["excerpt"]
        source_id, txt_sha256, start = source_map["source_id"], source_map["txt"]["sha256"], recorded["start"]
        units_recorded = source_map["units"]
    except (KeyError, TypeError):
        raise EvidenceError("Карта источника не содержит обязательных полей снимка") from None
    excerpt_sha256 = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    if (excerpt_sha256 != recorded.get("sha256") or recorded.get("end") != start + len(excerpt)
            or recorded.get("chars") != len(excerpt)):
        raise EvidenceError("Фрагмент не совпадает с хешем или границами карты источника")
    points = list(dict.fromkeys(unit["point"] for unit in units_recorded))
    units = tuple(line_units(excerpt, points))
    if [unit.__dict__ for unit in units] != units_recorded:
        raise EvidenceError("Карта строк не воспроизводится из фрагмента")
    canonical = {"contract": EVIDENCE_CONTRACT, "source_id": source_id, "txt_sha256": txt_sha256,
                 "excerpt_sha256": excerpt_sha256, "excerpt_start": start, "excerpt_end": start + len(excerpt),
                 "units": units_recorded}
    return Snapshot("snap-" + _digest(canonical)[:24], source_id, txt_sha256, excerpt_sha256, start, excerpt, units)


def resolve(snap: Snapshot, line_id: str, start: int | None = None, end: int | None = None) -> dict[str, Any]:
    """Доказательство — вся строка или непустой поддиапазон одной строки; смещения — во фрагменте."""
    unit = next((candidate for candidate in snap.units if candidate.id == line_id), None)
    if unit is None:
        raise EvidenceError("Строка с таким ID отсутствует в карте снимка")
    if (start is None) != (end is None):
        raise EvidenceError("Поддиапазон задаётся обеими границами start и end")
    start, end = (unit.start, unit.end) if start is None else (start, end)
    if not (isinstance(start, int) and isinstance(end, int)) or isinstance(start, bool) or isinstance(end, bool):
        raise EvidenceError("Границы доказательства должны быть целыми числами")
    if not unit.start <= start < end <= unit.end:
        raise EvidenceError("Диапазон пуст или выходит за границы одной строки")
    text = snap.excerpt[start:end]
    if not text.strip():
        raise EvidenceError("Доказательство не может состоять только из пробельных символов")
    record = {"contract": EVIDENCE_CONTRACT, "snapshot_id": snap.snapshot_id, "source_id": snap.source_id,
              "excerpt_sha256": snap.excerpt_sha256, "line_id": unit.id, "point": unit.point,
              "start": start, "end": end, "whole_line": (start, end) == (unit.start, unit.end),
              # Смещения полного TXT включают BOM (символ 0), как в карте LE-2.
              "full_txt_start": snap.excerpt_start + start, "full_txt_end": snap.excerpt_start + end,
              "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "chars": len(text)}
    # Роль, утверждение и толкование в идентичность не входят.
    identity = [EVIDENCE_CONTRACT, snap.snapshot_id, start, end]
    return {"evidence_id": "ev-" + _digest(identity)[:20], **record}


def text_of(snap: Snapshot, record: dict[str, Any]) -> str:
    """Перепроверить запись (без лишних полей, тот же ID и хеш) и вернуть текст, вырезанный из снимка."""
    if not isinstance(record, dict) or set(record) != EVIDENCE_FIELDS:
        raise EvidenceError("Запись доказательства содержит неизвестные или не содержит обязательных полей")
    if record["snapshot_id"] != snap.snapshot_id:
        raise EvidenceError("Доказательство относится к другому снимку")
    if record != resolve(snap, record["line_id"], record["start"], record["end"]):
        raise EvidenceError("Запись доказательства не воспроизводится из снимка (ID, границы или хеш)")
    return snap.excerpt[record["start"]:record["end"]]


def verify_full_txt(snap: Snapshot, full_text: str, records: Iterable[dict[str, Any]]) -> bool:
    """Необязательная сверка смещений полного TXT: срез полного текста равен срезу фрагмента."""
    if hashlib.sha256(full_text.encode("utf-8")).hexdigest() != snap.txt_sha256:
        raise EvidenceError("Полный TXT не совпадает с SHA-256 из карты источника")
    return all(full_text[r["full_txt_start"]:r["full_txt_end"]] == text_of(snap, r) for r in records)


# --- ContextPackage --------------------------------------------------------------------------------

def _code(value: Any, field: str) -> str:
    if not isinstance(value, str) or not NOTE_CODE.fullmatch(value):
        raise EvidenceError(f"{field}: нужен код из заглавной латиницы, цифр и '_'")
    return value


def _selection_entries(selection: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (not isinstance(selection, dict) or set(selection) != {"contract", "evidence", "unresolved"}
            or selection["contract"] != SELECTION_CONTRACT):
        raise EvidenceError("Манифест выбора: неизвестный контракт или неожиданный набор полей")
    entries, dependencies = selection["evidence"], selection["unresolved"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_PACKAGE_EVIDENCE:
        raise EvidenceError(f"Манифест выбора: нужно 1–{MAX_PACKAGE_EVIDENCE} доказательств")
    if not isinstance(dependencies, list) or len(dependencies) > MAX_DEPENDENCIES:
        raise EvidenceError(f"Манифест выбора: не более {MAX_DEPENDENCIES} неразрешённых зависимостей")
    for entry in entries:
        if not isinstance(entry, dict) or not {"line", "role"} <= set(entry) <= {"line", "role", "start", "end"}:
            raise EvidenceError("Элемент выбора: допустимы только поля line, role, start, end")
        if entry["role"] not in ROLES or not isinstance(entry["line"], str) or not LINE_ID.fullmatch(entry["line"]):
            raise EvidenceError("Элемент выбора: неизвестная роль или некорректный ID строки")
    for dependency in dependencies:
        if not isinstance(dependency, dict) or set(dependency) != {"kind", "line", "note"}:
            raise EvidenceError("Зависимость: нужны ровно поля kind, line, note")
        if dependency["kind"] not in DEPENDENCY_KINDS:
            raise EvidenceError("Зависимость: неизвестный вид")
        _code(dependency["note"], "Зависимость.note")
    return entries, dependencies


def derived_links(snap: Snapshot, included: Sequence[dict[str, Any]], references: Sequence[Any]) -> dict[str, Any]:
    """Связи из детерминированного разбора отсылок LE-2 (только ID строк и номера, без текста отсылки).

    Показывают, что связано с целью, но не включено, и чем «не включено» отличается от «не связано».
    Ни одна связь не доказывает юридического действия оговорки.
    """
    target = next(item["evidence"] for item in included if item["role"] == "TARGET")
    target_line = next(unit for unit in snap.units if unit.id == target["line_id"])
    included_lines = {item["evidence"]["line_id"] for item in included}
    points_in_excerpt = {unit.point for unit in snap.units}
    to_target, from_target = [], []
    for reference in references:
        source_line = next(unit for unit in snap.units if unit.start <= reference.start < unit.end + 1)
        entry = {"from_line": source_line.id, "point": reference.point, "abzacs": list(reference.abzacs),
                 "external": reference.external}
        if (not reference.external and reference.point == target_line.point and source_line.id != target_line.id
                and (not reference.abzacs or target_line.abzac in reference.abzacs)):
            to_target.append({**entry, "granularity": "LINE" if reference.abzacs else "WHOLE_POINT",
                              "included": source_line.id in included_lines})
        if source_line.id == target_line.id:
            in_excerpt = reference.point in points_in_excerpt and not reference.external
            lines = [unit.id for unit in snap.units if unit.point == reference.point] if in_excerpt else []
            from_target.append({**entry, "in_excerpt": in_excerpt,
                                "referenced_lines_not_included": [line for line in lines if line not in included_lines]})
    out_of_excerpt = sorted({(reference.point, unit.id) for reference in references
                             for unit in snap.units if unit.start <= reference.start < unit.end + 1
                             and unit.id in included_lines
                             and (reference.external or reference.point not in points_in_excerpt)})
    return {
        "method": "LE-2 references.parse_references; abzac задан только для пункта-перечня",
        "references_to_target_line": to_target,
        "references_from_target_line": from_target,
        "omitted_lines_referencing_target": sorted({link["from_line"] for link in to_target if not link["included"]}),
        "included_lines_with_out_of_excerpt_references": [{"line": line, "point": point}
                                                          for point, line in out_of_excerpt],
        "not_included_not_linked": [unit.id for unit in snap.units if unit.id not in included_lines
                                    and unit.id not in {link["from_line"] for link in to_target}
                                    and not any(unit.id in link["referenced_lines_not_included"]
                                                for link in from_target)],
    }


def _dependency_id(snap: Snapshot, dependency: dict[str, Any]) -> str:
    body = {"kind": dependency["kind"], "line": dependency["line"], "note": dependency["note"]}
    return "dep-" + _digest([snap.snapshot_id, body])[:12]


def _check_included(included: Sequence[dict[str, Any]], unresolved: Sequence[dict[str, Any]]) -> None:
    """Общие инварианты пакета при сборке и при повторной загрузке."""
    if not 1 <= len(included) <= MAX_PACKAGE_EVIDENCE or len(unresolved) > MAX_DEPENDENCIES:
        raise EvidenceError("Пакет вне пределов числа доказательств или зависимостей")
    records = [item["evidence"] for item in included]
    if len({record["evidence_id"] for record in records}) != len(records):
        raise EvidenceError("Одно доказательство выбрано повторно (дубль или конфликт ролей)")
    if any(left["end"] > right["start"] for left, right in zip(records, records[1:])):
        raise EvidenceError("Доказательства пакета перекрываются или не упорядочены; один символ — одна роль")
    if [item["role"] for item in included].count("TARGET") != 1:
        raise EvidenceError("В пакете должна быть ровно одна цель TARGET")
    if sum(record["chars"] for record in records) > MAX_PACKAGE_CHARS:
        raise EvidenceError(f"Пакет превышает предел {MAX_PACKAGE_CHARS} символов")
    lines = {record["line_id"] for record in records}
    if any(dependency["line"] not in lines for dependency in unresolved):
        raise EvidenceError("Зависимость должна ссылаться на строку, включённую в пакет")
    if len({dependency["dependency_id"] for dependency in unresolved}) != len(unresolved):
        raise EvidenceError("Неразрешённая зависимость указана повторно")


def build_package(snap: Snapshot, selection: Any, selection_sha256: str,
                  references: Sequence[Any] | None) -> dict[str, Any]:
    """Детерминированно собрать пакет из явного выбора; дубли, конфликты ролей и перекрытия — отказ."""
    entries, dependencies = _selection_entries(selection)
    included = sorted(({"evidence": resolve(snap, entry["line"], entry.get("start"), entry.get("end")),
                        "role": entry["role"]} for entry in entries), key=lambda item: item["evidence"]["start"])
    unresolved = []
    for dependency in dependencies:
        body = {"kind": dependency["kind"], "line": dependency["line"], "note": dependency["note"]}
        unresolved.append({"dependency_id": _dependency_id(snap, body), **body})
    _check_included(included, unresolved)
    package = {
        "contract": CONTEXT_CONTRACT, "snapshot": snap.identity(),
        "target": next(item["evidence"]["evidence_id"] for item in included if item["role"] == "TARGET"),
        "evidence": included, "unresolved_dependencies": unresolved,
        "derived_links": derived_links(snap, included, references) if references is not None else None,
        "context_completeness": CONTEXT_COMPLETENESS,
        "limits": dict(PACKAGE_LIMITS),
        "selection_sha256": selection_sha256, "legal_status": "UNVERIFIED", "review_status": "REVIEW_REQUIRED",
    }
    return {"context_package_id": "ctx-" + _digest(package)[:20], **package}


def check_package(snap: Snapshot, package: Any) -> list[tuple[dict[str, Any], str]]:
    """Перепроверить сохранённый пакет: поля, снимок, доказательства, ID зависимостей, limits и ID пакета.

    derived_links при загрузке не пересчитываются (это потребовало бы разбора отсылок LE-2 и LangExtract):
    это сохранённая диагностика сборки, а не проверенные факты. ID пакета подтверждает целостность, не подлинность.
    """
    if not isinstance(package, dict) or set(package) != PACKAGE_FIELDS or package["contract"] != CONTEXT_CONTRACT:
        raise EvidenceError("Пакет: неизвестный контракт или неожиданный набор полей")
    if package["snapshot"] != snap.identity():
        raise EvidenceError("Пакет собран для другого снимка")
    if (package["context_completeness"] != CONTEXT_COMPLETENESS or package["legal_status"] != "UNVERIFIED"
            or package["review_status"] != "REVIEW_REQUIRED"):
        raise EvidenceError("Пакет: недопустимые значения статусов")
    body = {key: value for key, value in package.items() if key != "context_package_id"}
    if package["context_package_id"] != "ctx-" + _digest(body)[:20]:
        raise EvidenceError("ID пакета не совпадает с содержимым")
    included, unresolved = package["evidence"], package["unresolved_dependencies"]
    if not isinstance(included, list) or not isinstance(unresolved, list):
        raise EvidenceError("Пакет: evidence и unresolved_dependencies должны быть списками")
    for item in included:
        if not isinstance(item, dict) or set(item) != {"evidence", "role"} or item["role"] not in ROLES:
            raise EvidenceError("Пакет: некорректный элемент доказательства")
        text_of(snap, item["evidence"])
    for dependency in unresolved:
        if (not isinstance(dependency, dict) or set(dependency) != {"dependency_id", "kind", "line", "note"}
                or dependency["kind"] not in DEPENDENCY_KINDS):
            raise EvidenceError("Пакет: некорректная неразрешённая зависимость")
        _code(dependency["note"], "Зависимость.note")
        # ID зависимости пересчитывается по правилу сборщика: совпадения хеша пакета недостаточно.
        if not LINE_ID.fullmatch(str(dependency["line"])) or dependency["dependency_id"] != _dependency_id(snap, dependency):
            raise EvidenceError("Пакет: ID неразрешённой зависимости не воспроизводится")
    limits = package["limits"]
    if (not isinstance(limits, dict) or set(limits) != set(PACKAGE_LIMITS)
            or any(type(limits[key]) is not type(value) or limits[key] != value for key, value in PACKAGE_LIMITS.items())):
        raise EvidenceError("Пакет: поле limits не совпадает с пределами реализации")
    _check_included(included, unresolved)
    target = next(item["evidence"]["evidence_id"] for item in included if item["role"] == "TARGET")
    if package["target"] != target:
        raise EvidenceError("Пакет: поле target не совпадает с доказательством роли TARGET")
    return [(item["evidence"], item["role"]) for item in included]
