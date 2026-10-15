"""Предварительная иерархия единиц (этап 1.2a.1-B): чистая логика без Docling и ввода-вывода.

Два разных механизма родительства:

- **структурная область** — документ → глава → статья. Глава открывает область, которая может
  пересекать домены нумерации (каждая статья открывает свой домен); область закрывается или
  блокируется явным перечнем границ, родитель через них не выводится;
- **номерной префикс** — статья → «N.» / «N-M.» и «N.» → «N.M.» → «N.M.K.». Связь «номер → номер»
  допустима только внутри одного домена нумерации 1.2a.1-A и только при точном непосредственном
  префиксе; «N.» → статья — только в домене, который открыт самой статьёй.

Родитель, который нельзя доказать, остаётся неизвестным с машинной причиной и требованием
рассмотрения. Юридический вид номера (пункт, часть, подпункт) не устанавливается.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Literal

from .boundaries import (ACT_TITLE, CHAPTER, CHAPTER_VARIANT, NUM_DOTTED, NUM_INSERTED, NUM_SINGLE,
                         QUOTED_MARKER, STRUCTURAL_LIKE, Boundary, NumberingDomain, NumberingMarker,
                         QuotedContext, article_match, marker_category)

Inner = Literal["none", "numbering", "structural", "uninspectable"]
Status = Literal["root", "resolved", "review_required"]

# «Статья N-M.» / «Статья N-M .» с разделителем или весь абзац «Статья N-M» — как правило статьи 1.2a.
ARTICLE_VARIANT_UNIT = re.compile(r"Статья ([1-9][0-9]{0,3}-[1-9][0-9]?)(?: ?\.(?=\s|$)|$)", re.IGNORECASE)
# Пролог документа: собственные название/заголовок/гриф до первой единицы или номера.
PRELUDE_TRANSPARENT = frozenset({"act_title_candidate", "unrecognized_heading", "approval_stamp"})
# Контейнеры, которые не являются единицами: закрывают главу и уровень документа.
CONTAINERS = frozenset({"section", "appendix", "approval_stamp", "act_title_candidate", "unrecognized_heading"})
# Неопределённые границы: уровень неизвестен, поэтому блокируют и главу, и уровень документа.
UNCERTAIN = frozenset({"unrecognized_structural_marker", "numbering_restart", "amendment_text_undelimited"})
# Закрытая цитата не создаёт контейнер, но её закрытие эвристично: область главы через неё не тянется.
QUOTE_EDGES = frozenset({"quoted_block_start", "quoted_block_end"})
ARTICLE_LEVEL = frozenset({"article", "article_variant"})
SEQUENCE_ANOMALIES = frozenset({"gap", "repeat", "backward", "inserted_unmatched"})

CODES: dict[str, str] = {
    "chapter_variant_marker": "изолированное «ГЛАВА N-M»; правовой статус вставки не установлен",
    "article_variant_marker": "«Статья N-M.» с разделителем или весь абзац «Статья N-M»; не тождественна статье N",
    "structural_scope": "позиционная структурная область без промежуточных контейнеров и неопределённых границ; "
                        "ancestry Docling ей не противоречит",
    "article_numbering_area": "номер верхнего уровня в домене, открытом подтверждённой статьёй-единицей",
    "number_prefix": "точный непосредственный префикс: ближайший предшествующий номер уровня выше в том же домене, "
                     "единственный",
    "container_not_modeled": "ближайшая контейнерная граница (раздел, приложение, гриф, название акта, заголовок) "
                             "не является единицей; непосредственный родитель не моделируется",
    "scope_uncertain": "между кандидатом в родители и единицей неопределённая граница (нераспознанный маркер, "
                       "сброс нумерации, цитата, текст изменения без кавычек); родитель не выводится",
    "barrier_not_transparent": "между кандидатом в родители и единицей stop_only, непроницаемый для этой связи: "
                               "структурный маркер в начале строки, непроверяемое содержимое или цитата; для номерной "
                               "связи также номер",
    "quoted_context": "единица внутри предполагаемой цитаты; не потомок исходного акта без независимого доказательства",
    "ancestry_position_conflict": "явная ancestry Docling не совпадает с позиционной областью",
    "no_article_area": "домен номера открыт не статьёй-единицей (глава, приложение, название, цитата, сброс и т. п.)",
    "sequence_anomaly": "номер верхнего уровня в аномалии последовательности домена или после неё",
    "prefix_parent_missing": "нет предшествующего номера с точным непосредственным префиксом в домене",
    "prefix_only_in_other_domain": "номер с нужным префиксом есть только в другом домене; домены не связываются",
    "prefix_parent_ambiguous": "номер с нужным префиксом встречается в домене больше одного раза",
    "prefix_parent_not_nearest": "ближайший предшествующий номер уровня выше имеет другое обозначение",
    "inserted_base_missing": "вставной «N.M-K.» без предшествующего «N.M» в домене",
    "crossed_inspected_barrier": "связь проходит через stop_only с проверенным содержимым: для «глава → статья» — "
                                 "без структурных маркеров (номера допустимы), для номерной связи — без маркеров и "
                                 "номеров; правильность связи не доказывает",
    "root": "корень",
    "resolved": "родитель установлен по правилу политики; предварительно, юридически не проверен",
    "review_required": "родитель не установлен; требуется ручное рассмотрение",
}


@dataclass(frozen=True)
class Candidate:
    """Единица до назначения родителя. `ancestry_*` — результат правил 1.2a по дереву Docling."""

    id: str
    kind: Literal["chapter", "article", "numbered"]
    number: str
    order: int
    source_refs: tuple[str, ...]
    anchor_id: str
    domain_id: str
    quoted_context: QuotedContext
    ancestry_parent: str | None = None
    ancestry_basis: str = "docling_ancestry"
    level: int | None = None
    sequence: str | None = None


@dataclass(frozen=True)
class Placement:
    parent_id: str | None
    parent_basis: str
    parent_status: Status
    parent_evidence: tuple[str, ...]
    crossed_barrier: bool = False


def unit_marker(cue: str) -> tuple[Literal["chapter", "article"], str, str] | None:
    """Вид, номер и основание якоря главы/статьи; варианты N-M — только с разделителем."""
    if match := CHAPTER.fullmatch(cue):
        return "chapter", match.group(1), "isolated_chapter_marker"
    if match := CHAPTER_VARIANT.fullmatch(cue):
        return "chapter", match.group(1), "chapter_variant_marker"
    if match := article_match(cue):
        return "article", match.group(1), "article_marker_with_delimiter"
    if match := ARTICLE_VARIANT_UNIT.match(cue):
        return "article", match.group(1), "article_variant_marker"
    return None


def classify_inner(texts: Iterable[str]) -> Inner:
    """Содержимое stop_only-элемента: структурный маркер важнее номера; пустое — «none».

    Маркер ищется в начале каждой логической строки каждого текста (`str.splitlines`: `\\n`, `\\r\\n`, ручной
    разрыв `\\v` и т. п.), а не только в начале склеенной ячейки. Ссылка внутри строки («см. статью 12») маркером
    не считается."""
    found: Inner = "none"
    for line in (line for text in texts for line in text.splitlines()):
        cue = re.sub(r"\s+", " ", line).strip()
        if not cue:
            continue
        letters = [char for char in cue if char.isalpha()]
        if (marker_category(cue) is not None or STRUCTURAL_LIKE.match(cue) or QUOTED_MARKER.match(cue)
                or (ACT_TITLE.match(cue) and all(char.isupper() for char in letters))):
            return "structural"
        if NUM_INSERTED.match(cue) or NUM_DOTTED.match(cue) or NUM_SINGLE.match(cue):
            found = "numbering"
    return found


def _resolved(parent: str, basis: str, evidence: tuple[str, ...], crossed: bool = False) -> Placement:
    return Placement(parent, basis, "resolved", evidence, crossed)


def _review(basis: str, evidence: tuple[str, ...] = ()) -> Placement:
    return Placement(None, basis, "review_required", evidence)


def place(candidates: Sequence[Candidate], root_id: str, boundaries: Sequence[Boundary],
          markers: Sequence[NumberingMarker], domains: Sequence[NumberingDomain],
          inner: Mapping[int, Inner], container: Mapping[str, str]) -> dict[str, Placement]:
    """Родители всех кандидатов. `inner` — содержимое stop_only по порядку потока, `container` —
    непосредственный родитель узла в дереве Docling."""
    by_anchor = {c.anchor_id: c for c in candidates}
    stops = [b for b in boundaries if b.effect == "stop_only"]
    result = _structural(candidates, root_id, boundaries, by_anchor, inner, markers)
    result.update(_numbered(candidates, by_anchor, stops, markers, domains, inner, container))
    return result


def _structural(candidates: Sequence[Candidate], root_id: str, boundaries: Sequence[Boundary],
                by_anchor: Mapping[str, Candidate], inner: Mapping[int, Inner],
                markers: Sequence[NumberingMarker]) -> dict[str, Placement]:
    """Состояние уровня документа и уровня главы вдоль границ потока."""
    result: dict[str, Placement] = {}
    first_marker = min((m.order for m in markers if m.verification == "typed_in_snapshot_text"
                        and m.quoted_context == "none"), default=None)
    prelude = True
    # Блокировка уровня: (основание, ID границы); None — уровень открыт.
    document_block: tuple[str, str] | None = None
    chapter: Candidate | None = None
    chapter_block: tuple[str, str] | None = None
    crossed = False

    def block(basis: str, boundary: Boundary, document_level: bool) -> None:
        nonlocal chapter, chapter_block, document_block, prelude
        chapter, chapter_block, prelude = None, (basis, boundary.id), False
        if document_level:
            document_block = (basis, boundary.id)

    def from_document() -> Placement:
        if document_block is not None:
            return _review(document_block[0], (document_block[1],))
        return _resolved(root_id, "structural_scope", ())

    for boundary in boundaries:
        if first_marker is not None and boundary.order > first_marker:
            prelude = False
        unit = by_anchor.get(boundary.id)
        if boundary.quoted_context != "none" and boundary.category != "quoted_block_start":
            if unit is not None:
                result[unit.id] = _review("quoted_context", (boundary.id,))
            continue
        if boundary.effect == "stop_only":
            if boundary.category == "quoted_marker" or inner.get(boundary.order) in {"structural", "uninspectable"}:
                block("barrier_not_transparent", boundary, True)
            else:
                crossed = crossed or chapter is not None
            continue
        if boundary.category in QUOTE_EDGES:
            # Цитата не вводит контейнер: уровень документа не меняется, открытая глава — блокируется.
            if chapter is not None:
                chapter, chapter_block = None, ("scope_uncertain", boundary.id)
            continue
        if unit is not None and unit.kind == "chapter":
            result[unit.id] = _with_ancestry(unit, from_document(), root_id)
            chapter, chapter_block, crossed, prelude = unit, None, False, False
            continue
        if unit is not None:
            if chapter is not None:
                placement = _resolved(chapter.id, "structural_scope", (chapter.anchor_id,), crossed)
            elif chapter_block is not None:
                placement = _review(chapter_block[0], (chapter_block[1],))
            else:
                placement = from_document()
            result[unit.id] = _with_ancestry(unit, placement, root_id)
            prelude = False
            continue
        if boundary.category in ARTICLE_LEVEL:
            # Статья без якоря-единицы — граница уровня статьи: область главы не закрывает.
            prelude = False
            continue
        if prelude and boundary.category in PRELUDE_TRANSPARENT:
            continue
        if boundary.category in CONTAINERS:
            block("container_not_modeled", boundary, True)
        elif boundary.category in UNCERTAIN:
            block("scope_uncertain", boundary, True)
        else:
            raise ValueError("Категория границы без правила 1.2a.1-B")
    missing = {c.id for c in candidates if c.kind != "numbered"} - result.keys()
    if missing:
        raise ValueError("Якорь главы/статьи без своей границы")
    return result


def _with_ancestry(unit: Candidate, placement: Placement, root_id: str) -> Placement:
    """Приоритет: цитата → неопределённая ancestry 1.2a → явная ancestry против позиции."""
    if unit.quoted_context != "none":
        return _review("quoted_context", (unit.anchor_id,))
    if unit.ancestry_parent is None:
        return _review(unit.ancestry_basis)
    if unit.ancestry_parent == root_id:
        return placement  # плоское дерево: ancestry до body не несёт вложенности
    if placement.parent_id == unit.ancestry_parent:
        return Placement(placement.parent_id, "docling_ancestry", "resolved", placement.parent_evidence,
                         placement.crossed_barrier)
    return _review("ancestry_position_conflict", placement.parent_evidence)


def _numbered(candidates: Sequence[Candidate], by_anchor: Mapping[str, Candidate], stops: Sequence[Boundary],
              markers: Sequence[NumberingMarker], domains: Sequence[NumberingDomain],
              inner: Mapping[int, Inner], container: Mapping[str, str]) -> dict[str, Placement]:
    result: dict[str, Placement] = {}
    by_domain = {d.id: d for d in domains}
    openers = {d.id: d.opened_by for d in domains}
    seen: dict[str, list[Candidate]] = {}
    designations_seen: set[str] = set()
    anomaly: dict[str, str] = {}

    def barrier(start: int, end: int) -> tuple[Placement | None, bool]:
        crossed = False
        for stop in stops:
            if start < stop.order < end:
                if stop.category == "quoted_marker" or inner.get(stop.order, "uninspectable") != "none":
                    return _review("barrier_not_transparent", (stop.id,)), False
                crossed = True
        return None, crossed

    def same_container(child: Candidate, parent: Candidate) -> bool:
        return container.get(child.source_refs[0]) in {container.get(parent.source_refs[0]), parent.source_refs[0]}

    for unit in (c for c in candidates if c.kind == "numbered"):
        prior = seen.setdefault(unit.domain_id, [])
        result[unit.id] = _number_parent(unit, prior, by_domain, openers, by_anchor, anomaly,
                                         designations_seen, barrier, same_container)
        prior.append(unit)
        designations_seen.add(unit.number)
    return result


def _number_parent(unit: Candidate, prior: list[Candidate], by_domain: Mapping[str, NumberingDomain],
                   openers: Mapping[str, str | None], by_anchor: Mapping[str, Candidate],
                   anomaly: dict[str, str], designations_seen: set[str],
                   barrier: Callable[[int, int], tuple[Placement | None, bool]],
                   same_container: Callable[[Candidate, Candidate], bool]) -> Placement:
    if unit.quoted_context != "none":
        return _review("quoted_context", (unit.anchor_id,))
    domain = by_domain[unit.domain_id]
    if unit.level == 1:
        opener = openers[unit.domain_id]
        article = by_anchor.get(opener) if opener else None
        if article is None or article.kind != "article" or article.quoted_context != "none":
            return _review("no_article_area", (domain.id,) + ((opener,) if opener else ()))
        if unit.sequence in SEQUENCE_ANOMALIES:
            anomaly.setdefault(domain.id, unit.anchor_id)
        if domain.id in anomaly:
            return _review("sequence_anomaly", (anomaly[domain.id],))
        parent = article
        basis = "article_numbering_area"
    else:
        parts = unit.number.split(".")
        wanted = ".".join(parts[:-1])
        nearest = next((p for p in reversed(prior) if (p.level or 0) < (unit.level or 0)), None)
        same = [p for p in prior if p.number == wanted]
        if not same:
            return _review("prefix_only_in_other_domain" if wanted in designations_seen else "prefix_parent_missing")
        if len(same) > 1:
            return _review("prefix_parent_ambiguous", tuple(p.anchor_id for p in same))
        if nearest is not same[0]:
            return _review("prefix_parent_not_nearest", tuple(p.anchor_id for p in (nearest, same[0]) if p))
        if "-" in parts[-1]:
            base = ".".join([*parts[:-1], parts[-1].split("-")[0]])
            if not any(p.number == base for p in prior):
                return _review("inserted_base_missing", (same[0].anchor_id,))
        parent = same[0]
        basis = "number_prefix"
    blocked, crossed = barrier(parent.order, unit.order)
    if blocked is not None:
        return blocked
    if not same_container(unit, parent):
        return _review("ancestry_position_conflict", (parent.anchor_id,))
    return _resolved(parent.id, basis, (domain.id, parent.anchor_id), crossed)
