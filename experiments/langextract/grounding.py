"""Независимая проверка привязки, учёт сырого вывода и разрешение кандидатов связей (без ввода-вывода).

Вердикт привязки не зависит от alignment_status LangExtract: выравнивание библиотеки
токенное и без учёта регистра, а нечёткие совпадения тоже получают смещения. Проверенным
считается только диапазон, для которого source_text[start:end] == extraction_text.
"""

from collections import Counter
from collections.abc import Sequence
import hashlib
import re
from typing import Any

from llm_wiki.validation import parse_json

from . import fixture
from .task import RELATION_TYPES, TARGET_REFERENCE

VERIFIED = "VERIFIED_EXACT_SPAN"
NO_POSITION = "NO_POSITION"
MISMATCH = "OFFSET_TEXT_MISMATCH"
INVALID_INTERVAL = "INVALID_INTERVAL"
EMPTY_TEXT = "EMPTY_TEXT"
LABEL_PREFIX = re.compile(r"(\d{1,4})\. ")


def paragraphs(text: str) -> list[tuple[int, int]]:
    """Абзацы входного TXT: непустые участки между пустыми строками (политика формата фикстуры)."""
    spans, offset = [], 0
    for segment in text.split("\n\n"):
        stripped = segment.strip("\n")
        if stripped.strip():
            start = offset + segment.index(stripped)
            spans.append((start, start + len(stripped)))
        offset += len(segment) + 2
    return spans


def stable_id(kind: str, source_sha256: str, start: int, end: int, extraction_class: str) -> str:
    """Экспериментальный ID только для проверенного диапазона: источник + диапазон + класс."""
    digest = hashlib.sha256(f"{source_sha256}:{start}:{end}:{extraction_class}".encode()).hexdigest()
    return f"le1-{kind}-{digest[:16]}"


def _occurrences(text: str, needle: str) -> list[int]:
    found, start = [], text.find(needle)
    while needle and start >= 0:
        found.append(start)
        start = text.find(needle, start + 1)
    return found


def _non_verbatim_attributes(extraction_text: str, attributes: dict[str, Any] | None) -> list[str]:
    result = []
    for key, value in (attributes or {}).items():
        values = value if isinstance(value, list) else [value]
        if any(not isinstance(item, str) or item not in extraction_text for item in values):
            result.append(key)
    return sorted(result)


def ground(text: str, source_sha256: str, extractions: Sequence[Any], kind: str) -> list[dict[str, Any]]:
    """Проверить каждую итоговую Extraction и назначить ID только проверенным диапазонам."""
    paras = paragraphs(text)
    items: list[dict[str, Any]] = []
    first_by_id: dict[str, int] = {}
    for position, extraction in enumerate(extractions):
        value = extraction.extraction_text
        interval = extraction.char_interval
        start = interval.start_pos if interval is not None else None
        end = interval.end_pos if interval is not None else None
        occurrences = _occurrences(text, value)
        if not value:
            status = EMPTY_TEXT
        elif start is None or end is None:
            status = NO_POSITION
        elif not 0 <= start < end <= len(text):
            status = INVALID_INTERVAL
        elif text[start:end] == value:
            status = VERIFIED
        else:
            status = MISMATCH
        item: dict[str, Any] = {
            "position": position,
            "class": extraction.extraction_class,
            "extraction_text": value,
            "attributes": extraction.attributes,
            "library_alignment_status": (extraction.alignment_status.value
                                         if extraction.alignment_status is not None else None),
            "char_interval": None if start is None and end is None else [start, end],
            "grounding": status,
            "source_slice_at_interval": (text[start:end] if status == MISMATCH else None),
            "exact_occurrences": len(occurrences),
            "library_offset_is_exact_occurrence": start in occurrences if start is not None else None,
            "ambiguous_text": len(occurrences) > 1,
            "id": None, "duplicate_of": None, "paragraphs": [], "paragraph_label": None,
            "non_verbatim_attributes": _non_verbatim_attributes(value, extraction.attributes),
        }
        if status == VERIFIED:
            touched = [index for index, (p_start, p_end) in enumerate(paras) if start < p_end and p_start < end]
            item["paragraphs"] = touched
            if len(touched) == 1:
                match = LABEL_PREFIX.match(text, paras[touched[0]][0])
                item["paragraph_label"] = match[1] if match else None
            identifier = stable_id(kind, source_sha256, start, end, item["class"])
            item["id"] = identifier
            if identifier in first_by_id:
                item["duplicate_of"] = first_by_id[identifier]
            else:
                first_by_id[identifier] = position
        items.append(item)
    return items


def overlaps(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Пары проверенных диапазонов, которые пересекаются, но не совпадают полностью."""
    verified = [item for item in items if item["grounding"] == VERIFIED]
    result = []
    for index, left in enumerate(verified):
        for right in verified[index + 1:]:
            (l_start, l_end), (r_start, r_end) = left["char_interval"], right["char_interval"]
            if l_start < r_end and r_start < l_end and (l_start, l_end) != (r_start, r_end):
                result.append({"positions": [left["position"], right["position"]],
                               "intervals": [[l_start, l_end], [r_start, r_end]]})
    return result


def same_span_different_class(items: Sequence[dict[str, Any]]) -> list[list[int]]:
    spans: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in items:
        if item["grounding"] == VERIFIED:
            spans.setdefault(tuple(item["char_interval"]), []).append(item)
    return [[item["position"] for item in group] for group in spans.values()
            if len({item["class"] for item in group}) > 1]


def raw_items(content: str, allowed_classes: Sequence[str]) -> dict[str, Any]:
    """Строгий разбор сырого ответа: повторные ключи, обёртки и неожиданные поля не исправляются."""
    try:
        parsed = parse_json(content)
    except ValueError as error:
        return {"strict_parse": "FAILED", "error": str(error), "items": []}
    issues, items = [], []
    if not isinstance(parsed, dict) or not isinstance(parsed.get("extractions"), list):
        return {"strict_parse": "OK", "error": "нет объекта с ключом extractions", "items": []}
    if set(parsed) != {"extractions"}:
        issues.append("extra_top_level_keys")
    for index, group in enumerate(parsed["extractions"]):
        if not isinstance(group, dict):
            issues.append(f"item_{index}_not_object")
            continue
        for key, value in group.items():
            if key.endswith("_attributes"):
                if key[:-len("_attributes")] not in group:
                    issues.append(f"item_{index}_orphan_attributes")
                if value is not None and not isinstance(value, dict):
                    issues.append(f"item_{index}_attributes_not_object")
                continue
            if key not in allowed_classes:
                issues.append(f"item_{index}_unknown_class")
            if not isinstance(value, str):
                issues.append(f"item_{index}_text_not_string")
            items.append({"class": key, "text": value if isinstance(value, str) else str(value),
                          "attributes": group.get(f"{key}_attributes")})
        if sum(not key.endswith("_attributes") for key in group) != 1:
            issues.append(f"item_{index}_class_count_not_1")
    return {"strict_parse": "OK", "issues": issues, "items": items}


def raw_vs_final(raw_per_call: Sequence[dict[str, Any]], items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Сопоставить мультимножества (класс, текст) сырых ответов и итоговых Extraction задачи."""
    raw = Counter((item["class"], item["text"]) for call in raw_per_call for item in call["items"])
    final = Counter((item["class"], item["extraction_text"]) for item in items)
    return {
        "raw_item_count": sum(raw.values()),
        "final_item_count": sum(final.values()),
        "absent_from_final": [{"class": c, "text": t, "count": n} for (c, t), n in (raw - final).items()],
        "not_in_raw": [{"class": c, "text": t, "count": n} for (c, t), n in (final - raw).items()],
    }


def fixture_coverage(text: str, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Метрики по эталонной разметке фикстуры; модели эта разметка не передаётся."""
    provisions = fixture.provisions(text)
    verified = [item for item in items if item["grounding"] == VERIFIED and item["duplicate_of"] is None]
    per_provision: dict[str, list[dict[str, Any]]] = {p.label: [] for p in provisions}
    merged, outside = [], []
    for item in verified:
        start, end = item["char_interval"]
        hits = [p.label for p in provisions if start < p.paragraph_end and p.paragraph_start < end]
        if len(hits) > 1:
            merged.append({"position": item["position"], "provisions": hits})
        if not hits:
            outside.append(item["position"])
        for label in hits:
            per_provision[label].append({"id": item["id"], "class": item["class"],
                                         "covers_whole_body": (start, end) == (
                                             next(p.body_start for p in provisions if p.label == label),
                                             next(p.paragraph_end for p in provisions if p.label == label))})
    ids_6 = {entry["id"] for entry in per_provision.get("6", [])}
    ids_7 = {entry["id"] for entry in per_provision.get("7", [])}
    return {
        "provisions": per_provision,
        "provisions_with_verified_passage": sum(bool(v) for v in per_provision.values()),
        "provisions_total": len(provisions),
        "missing_provisions": [label for label, value in per_provision.items() if not value],
        "passages_spanning_several_provisions": merged,
        "verified_passages_outside_provisions": outside,
        "provision_5_classes": sorted({entry["class"] for entry in per_provision.get("5", [])}),
        "provisions_6_7_distinct": bool(ids_6) and bool(ids_7) and not ids_6 & ids_7,
    }


def resolve_relations(text: str, relations: Sequence[dict[str, Any]],
                      passages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Источник — проверенный фрагмент LE-1A, содержащий доказательство; цель — по дословной отсылке «пункта N».

    Поиск цели — узкая экспериментальная проверка: абзац TXT, начинающийся с «N. ». Это не
    восстановление иерархии и не правовой разбор ссылок. Каждая связь остаётся REVIEW_REQUIRED.
    """
    paras = paragraphs(text)
    anchored = [item for item in passages if item["grounding"] == VERIFIED and item["duplicate_of"] is None]
    result = []
    for relation in relations:
        attributes = relation["attributes"] or {}
        reference = attributes.get("target_reference")
        relation_type = attributes.get("relation_type")
        entry: dict[str, Any] = {
            "position": relation["position"], "evidence_text": relation["extraction_text"],
            "evidence_grounding": relation["grounding"], "evidence_interval": relation["char_interval"],
            "relation_type": relation_type if relation_type in RELATION_TYPES else None,
            "raw_relation_type": relation_type, "target_reference": reference,
            "source_passage_ids": [], "source_label": None, "target_label": None,
            "target_passage_ids": [], "problems": [], "review_status": "REVIEW_REQUIRED",
        }
        problems = entry["problems"]
        if relation_type not in RELATION_TYPES:
            problems.append("RELATION_TYPE_UNKNOWN")
        if relation["grounding"] != VERIFIED:
            problems.append("EVIDENCE_NOT_GROUNDED")
        else:
            start, end = relation["char_interval"]
            sources = [p for p in anchored if p["char_interval"][0] <= start and end <= p["char_interval"][1]]
            entry["source_passage_ids"] = [p["id"] for p in sources]
            labels = {p["paragraph_label"] for p in sources}
            entry["source_label"] = labels.pop() if len(labels) == 1 else None
            if not sources:
                problems.append("SOURCE_PASSAGE_NOT_ANCHORED")
        if not isinstance(reference, str) or reference not in relation["extraction_text"]:
            problems.append("TARGET_REFERENCE_NOT_IN_EVIDENCE")
        match = TARGET_REFERENCE.fullmatch(reference) if isinstance(reference, str) else None
        if match is None:
            problems.append("TARGET_REFERENCE_UNPARSED")
        else:
            label = match[1]
            entry["target_label"] = label
            target_paras = [span for span in paras if text.startswith(f"{label}. ", span[0])]
            if not target_paras:
                problems.append("TARGET_PROVISION_MISSING")
            elif len(target_paras) > 1:
                problems.append("TARGET_PROVISION_AMBIGUOUS")
            else:
                p_start, p_end = target_paras[0]
                targets = [p for p in anchored if p_start <= p["char_interval"][0] and p["char_interval"][1] <= p_end]
                entry["target_passage_ids"] = [p["id"] for p in targets]
                if not targets:
                    problems.append("TARGET_PROVISION_WITHOUT_ANCHORED_PASSAGE")
                if entry["source_label"] == label:
                    problems.append("SELF_REFERENCE")
        entry["resolution"] = "CANDIDATE_RESOLVED" if not problems else "UNRESOLVED"
        result.append(entry)
    return result


def expected_link(relations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Оценка по фикстуре: найдена ли разрешённая связь 5 → 1 типа EXCEPTION_TO."""
    matching = [r for r in relations if r["resolution"] == "CANDIDATE_RESOLVED"
                and r["source_label"] == fixture.EXPECTED_EXCEPTION_SOURCE
                and r["target_label"] == fixture.EXPECTED_EXCEPTION_TARGET]
    reversed_ = [r for r in relations if r["source_label"] == fixture.EXPECTED_EXCEPTION_TARGET
                 and r["target_label"] == fixture.EXPECTED_EXCEPTION_SOURCE]
    return {
        "found_resolved_5_to_1": bool(matching),
        "relation_type_exception_to": any(r["relation_type"] == "EXCEPTION_TO" for r in matching),
        "reversed_candidates": len(reversed_),
        "other_candidates": len(relations) - len(matching) - len(reversed_),
    }
