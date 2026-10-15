"""Привязка к строкам пунктов, узкий разбор отсылок и разрешение кандидатов связей LE-2B (без ввода-вывода).

Разбираются только формы «пункте N», «абзаце <порядковое> пункта N», «абзацах <порядковое> и
<порядковое> пункта N» и отсылки к статьям кодексов (внешние). Всё остальное — UNPARSED.
Цель ищется только внутри фрагмента: пункт вне фрагмента не досматривается в полном TXT.
Ни один статус не означает юридической корректности; каждая связь — REVIEW_REQUIRED.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Any

from experiments.langextract.grounding import VERIFIED
from experiments.langextract.task import RELATION_TYPES

from .source import Line

ORDINALS = {"перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7,
            "восьм": 8, "девят": 9, "десят": 10}
_ORDINAL = r"(?:перв|втор|трет|четверт|пят|шест|седьм|восьм|девят|десят)[а-я]*"
REFERENCE = re.compile(
    rf"(?<![а-яё])(?:абзац[а-я]*\s+(?P<abzacs>{_ORDINAL}(?:(?:\s*,\s*|\s+и\s+){_ORDINAL})*)\s+)?"
    rf"пункт[а-я]*\s+(?P<point>\d{{1,4}})(?!\d|\.\d)(?P<tail>(?:\s+[^\s,;.:()]+){{0,3}})",
    re.IGNORECASE,
)
EXTERNAL_MARKERS = re.compile(r"(?<![а-яё])(?:стать[а-я]*|кодекс[а-я]*|закон[а-я]*|указ[а-я]*|декрет[а-я]*)(?![а-яё])",
                              re.IGNORECASE)
ABZAC_ASSUMPTION = "ABZAC_N_IS_LINE_N_OF_ENUMERATION_POINT"


@dataclass(frozen=True)
class Reference:
    raw: str
    start: int
    end: int
    point: str
    abzacs: tuple[int, ...]
    external: bool
    explicit_same_act: bool


def parse_references(text: str) -> list[Reference]:
    result = []
    for match in REFERENCE.finditer(text):
        abzacs = tuple(ORDINALS[stem] for word in re.findall(_ORDINAL, match["abzacs"] or "", re.IGNORECASE)
                       for stem in ORDINALS if word.lower().startswith(stem))
        tail = match["tail"] or ""
        result.append(Reference(match[0].rstrip(), match.start(), match.start() + len(match[0].rstrip()),
                                match["point"], abzacs, bool(EXTERNAL_MARKERS.search(tail)),
                                "настоящего положения" in tail.lower()))
    return result


def lines_for(units: Sequence[Line], start: int, end: int) -> list[Line]:
    return [unit for unit in units if unit.start < end and start < unit.end]


def occurrences(text: str, needle: str) -> list[int]:
    found, position = [], text.find(needle)
    while needle and position >= 0:
        found.append(position)
        position = text.find(needle, position + 1)
    return found


def attach_units(items: list[dict[str, Any]], excerpt: str, units: Sequence[Line], excerpt_offset: int) -> None:
    """Заменить поля абзацев LE-1 (разбиение по '\\n\\n' даёт здесь один «абзац 27») картой строк LE-2."""
    for item in items:
        item.pop("paragraphs", None)
        item.pop("paragraph_label", None)
        item["lines"], item["points"], item["full_txt_interval"] = [], [], None
        item["occurrence_lines"] = []
        if item["grounding"] != VERIFIED:
            item["source_attribution"] = "NOT_GROUNDED"
            continue
        start, end = item["char_interval"]
        touched = lines_for(units, start, end)
        item["lines"] = [unit.id for unit in touched]
        item["points"] = sorted({unit.point for unit in touched}, key=int)
        item["full_txt_interval"] = [start + excerpt_offset, end + excerpt_offset]
        item["occurrence_lines"] = [[unit.id for unit in lines_for(units, at, at + len(item["extraction_text"]))]
                                    for at in occurrences(excerpt, item["extraction_text"])]
        item["source_attribution"] = ("OCCURRENCE_DEPENDENT" if item["exact_occurrences"] > 1
                                      else "SPANS_SEVERAL_LINES" if len(touched) > 1 else "UNIQUE_LINE")


def _target_lines(units: Sequence[Line], point: str, abzacs: tuple[int, ...]) -> tuple[list[Line], list[str]]:
    in_point = [unit for unit in units if unit.point == point]
    if not abzacs:
        return in_point, []
    if any(unit.abzac is None for unit in in_point):
        return [], ["ABZAC_NUMBERING_UNSUPPORTED_FOR_POINT"]
    chosen = [unit for unit in in_point if unit.abzac in abzacs]
    return (chosen, []) if len(chosen) == len(abzacs) else ([], ["ABZAC_NOT_IN_POINT"])


def resolve(relation: dict[str, Any], units: Sequence[Line]) -> dict[str, Any]:
    """Источник — строки, содержащие проверенное доказательство; цель — по дословной отсылке и только во фрагменте."""
    attributes = relation.get("attributes") or {}
    raw_target = attributes.get("target_reference")
    raw_type = attributes.get("relation_type")
    evidence = relation["extraction_text"]
    entry: dict[str, Any] = {
        "position": relation["position"], "id": relation["id"], "evidence_text": evidence,
        "evidence_grounding": relation["grounding"], "evidence_interval": relation["char_interval"],
        "full_txt_interval": relation.get("full_txt_interval"),
        "source_attribution": relation.get("source_attribution"), "occurrence_lines": relation.get("occurrence_lines"),
        "source_lines": relation.get("lines", []), "source_point": None,
        "relation_type": raw_type if raw_type in RELATION_TYPES else None, "raw_relation_type": raw_type,
        "raw_target_reference": raw_target, "parsed_target": None, "references_in_evidence": [],
        "target_scope": None, "requested_granularity": None, "resolved_granularity": None,
        "target_lines": [], "assumptions": [], "problems": [], "resolution": None,
        "legal_status": "UNVERIFIED", "review_status": "REVIEW_REQUIRED",
    }
    problems = entry["problems"]
    if raw_type not in RELATION_TYPES:
        problems.append("RELATION_TYPE_UNKNOWN")
    in_evidence = parse_references(evidence)
    entry["references_in_evidence"] = [ref.__dict__ for ref in in_evidence]
    if relation["grounding"] != VERIFIED:
        problems.append("EVIDENCE_NOT_GROUNDED")
    else:
        points = relation.get("points", [])
        entry["source_point"] = points[0] if len(points) == 1 else None
        if len(points) != 1:
            problems.append("SOURCE_SPANS_SEVERAL_POINTS")
        if relation.get("source_attribution") == "OCCURRENCE_DEPENDENT":
            problems.append("SOURCE_OCCURRENCE_AMBIGUOUS")
    if not isinstance(raw_target, str) or not raw_target or raw_target not in evidence:
        problems.append("TARGET_REFERENCE_NOT_IN_EVIDENCE")
    parsed = parse_references(raw_target) if isinstance(raw_target, str) else []
    if len(parsed) != 1:
        problems.append("TARGET_REFERENCE_UNPARSED" if not parsed else "TARGET_REFERENCE_MULTIPLE")
        entry["resolution"] = "UNRESOLVED"
        return entry
    target = parsed[0]
    entry["parsed_target"] = target.__dict__
    unsupported = {"EVIDENCE_NOT_GROUNDED", "TARGET_REFERENCE_NOT_IN_EVIDENCE"} & set(problems)
    entry["requested_granularity"] = "abzac" if target.abzacs else "point"
    # Отсылка модели не должна быть грубее отсылки в самом доказательстве: «абзац 3 пункта 27» ≠ «пункт 27».
    finer = [ref for ref in in_evidence if ref.point == target.point and not ref.external
             and set(ref.abzacs) - set(target.abzacs)]
    if finer:
        problems.append("TARGET_LESS_PRECISE_THAN_EVIDENCE")
    if target.external:
        entry["target_scope"] = "EXTERNAL_ACT"
        problems.append("TARGET_OUTSIDE_EXCERPT")
        entry["resolution"] = "UNRESOLVED" if unsupported else "OUT_OF_SCOPE"
        return entry
    entry["target_scope"] = "SAME_ACT_EXPLICIT" if target.explicit_same_act else "SAME_ACT_ASSUMED"
    if not target.explicit_same_act:
        problems.append("SAME_ACT_NOT_EXPLICIT")
    if target.point not in {unit.point for unit in units}:
        problems.append("TARGET_OUTSIDE_EXCERPT")
        entry["resolution"] = "UNRESOLVED" if unsupported else "OUT_OF_SCOPE"
        return entry
    if entry["source_point"] == target.point:
        problems.append("SELF_REFERENCE")
    lines, issues = _target_lines(units, target.point, target.abzacs)
    problems.extend(issues)
    if lines:
        entry["target_lines"] = [unit.id for unit in lines]
        entry["resolved_granularity"] = "abzac" if target.abzacs else "point"
        if target.abzacs:
            entry["assumptions"].append(ABZAC_ASSUMPTION)
    else:
        entry["target_lines"] = [unit.id for unit in units if unit.point == target.point]
        entry["resolved_granularity"] = "point"
    blocking = [p for p in problems if p != "SAME_ACT_NOT_EXPLICIT"]
    if not blocking and entry["resolved_granularity"] == entry["requested_granularity"]:
        entry["resolution"] = "CANDIDATE_RESOLVED"
    elif entry["resolved_granularity"] and not unsupported:
        entry["resolution"] = "PARTIAL"
    else:
        entry["resolution"] = "UNRESOLVED"
    return entry


def evaluate_passages(checklist: dict[str, Any], items: Sequence[dict[str, Any]], excerpt: str,
                      units: Sequence[Line]) -> dict[str, Any]:
    """Покрытие строк проверенными фрагментами и классы против частного чек-листа (не полной разметки)."""
    verified = [item for item in items if item["grounding"] == VERIFIED and item["duplicate_of"] is None]
    expected = checklist["passages"]
    per_line = {}
    for unit in units:
        body_start = unit.start + (len(m[0]) if (m := re.match(r"\d{1,3}\. ", excerpt[unit.start:unit.end])) else 0)
        covered = set()
        hits = []
        for item in verified:
            start, end = item["char_interval"]
            if start < unit.end and unit.start < end:
                covered.update(range(max(start, body_start), min(end, unit.end)))
                hits.append({"position": item["position"], "class": item["class"],
                             "interval": item["char_interval"], "spans_lines": item["lines"]})
        spec = expected.get(unit.id, {})
        classes = [hit["class"] for hit in hits]
        acceptable = spec.get("acceptable_classes")
        per_line[unit.id] = {
            "verified_passages": hits,
            "body_coverage": round(len(covered) / max(1, unit.end - body_start), 3),
            "expected_min_parts": spec.get("min_parts", 0),
            "parts_found": len(hits),
            "classes": classes,
            "acceptable_classes": acceptable,
            "class_ok": None if not acceptable or not hits else all(c in acceptable for c in classes),
        }
    required = [line for line, spec in expected.items() if spec.get("min_parts", 0) > 0]
    return {
        "lines": per_line,
        "required_lines": len(required),
        "required_lines_covered": sum(per_line[line]["parts_found"] > 0 for line in required),
        "missing_required_lines": [line for line in required if per_line[line]["parts_found"] == 0],
        "lines_with_fewer_parts_than_expected": [line for line in required
                                                 if 0 < per_line[line]["parts_found"] < expected[line]["min_parts"]],
        "passages_spanning_several_lines": [item["position"] for item in verified if len(item["lines"]) > 1],
        "passages_spanning_several_points": [item["position"] for item in verified if len(item["points"]) > 1],
        "class_mismatches": [line for line, value in per_line.items() if value["class_ok"] is False],
        "uncovered_text_lines": [line for line, value in per_line.items() if value["body_coverage"] < 0.95],
    }


def evaluate_relations(checklist: dict[str, Any], resolved: Sequence[dict[str, Any]],
                       units: Sequence[Line]) -> dict[str, Any]:
    """Сопоставить кандидатов с ожиданиями чек-листа по (строка-источник, пункт-цель, абзацы)."""
    expectations = checklist["relations"]
    excerpt_points = {unit.point for unit in units}
    matched: dict[str, list[int]] = {exp["id"]: [] for exp in expectations}
    coarser: dict[str, list[int]] = {exp["id"]: [] for exp in expectations}
    unmatched = []

    def same_place(exp: dict[str, Any], entry: dict[str, Any], target: dict[str, Any]) -> bool:
        return (bool(set(exp["source_lines"]) & set(entry["source_lines"] or []))
                and exp["target_point"] == target["point"] and exp["external"] == target["external"])

    for entry in resolved:
        target = entry["parsed_target"]
        exact = [exp for exp in expectations if target is not None and same_place(exp, entry, target)
                 and sorted(exp["target_abzacs"]) == sorted(target["abzacs"])]
        # Та же строка и пункт, но другие абзацы (обычно огрублённая отсылка) — не засчитывается как найденное.
        near = [exp for exp in expectations if target is not None and same_place(exp, entry, target)]
        if exact:
            matched[exact[0]["id"]].append(entry["position"])
        elif near:
            coarser[near[0]["id"]].append(entry["position"])
        else:
            unmatched.append(entry["position"])
    by_position = {entry["position"]: entry for entry in resolved}
    rows = []
    for exp in expectations:
        found = [by_position[p] for p in matched[exp["id"]]]
        types = sorted({str(e["relation_type"]) for e in found})
        rows.append({
            "id": exp["id"], "required": exp["required"], "expected_type": exp["expected_type"],
            "found_positions": matched[exp["id"]], "wrong_granularity_positions": coarser[exp["id"]],
            "model_types": types,
            "type_ok": (None if not found or exp["expected_type"] == "AMBIGUOUS"
                        else types == [exp["expected_type"]]),
            "resolutions": sorted({e["resolution"] for e in found}),
            "resolved_as_expected": any(e["resolution"] == ("OUT_OF_SCOPE" if exp["external"] or exp["target_point"]
                                                            not in excerpt_points else "CANDIDATE_RESOLVED")
                                        for e in found),
        })
    return {
        "expectations": rows,
        "required_found": sum(bool(r["found_positions"]) for r in rows if r["required"]),
        "required_total": sum(r["required"] for r in rows),
        "false_negatives_required": [r["id"] for r in rows if r["required"] and not r["found_positions"]],
        "optional_not_found": [r["id"] for r in rows if not r["required"] and not r["found_positions"]],
        "found_only_with_wrong_granularity": [r["id"] for r in rows
                                              if r["wrong_granularity_positions"] and not r["found_positions"]],
        "type_errors": [r["id"] for r in rows if r["type_ok"] is False],
        "unmatched_candidates": unmatched,
        "resolution_counts": {status: sum(e["resolution"] == status for e in resolved)
                              for status in ("CANDIDATE_RESOLVED", "PARTIAL", "OUT_OF_SCOPE", "UNRESOLVED")},
    }
