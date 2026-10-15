"""Структурные границы и домены нумерации (этап 1.2a.1-A): чистая логика без Docling и ввода-вывода.

Термины контракта `legal-structure/v2`:

- маркер — совпадение узкого шаблона в начале текста элемента потока;
- граница — позиция потока: `stop_only` запрещает будущее присоединение через неё, не меняя
  нумерацию; `domain_boundary` закрывает текущий домен нумерации и открывает следующий;
- домен нумерации — интервал потока между соседними доменными границами, в котором набранные
  номера сравниваются между собой. Это область сопоставления номеров, **не** содержание
  правовой единицы: хвост домена может содержать подписи, реквизиты и посторонний текст.

Модуль не назначает родителей (это `hierarchy`, этап 1.2a.1-B), не присоединяет абзацы и не толкует
юридический смысл.
Номера вычисленного маркера Docling (`ListItem.marker`) не используются.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any, Literal

# Правила глав/статей этапа 1.2a, без изменений: единицы v2 строятся по ним же.
CHAPTER = re.compile(r"ГЛАВА ([1-9][0-9]{0,3})", re.IGNORECASE)
ARTICLE = re.compile(r"Статья ([1-9][0-9]{0,3})(?:\.|$)", re.IGNORECASE)
# Цифры только ASCII: \d и isdigit() принимают надстрочные и иные цифры Unicode.
_N = r"[1-9][0-9]{0,3}"
_SUFFIX = r"-[1-9][0-9]?"
CHAPTER_VARIANT = re.compile(rf"ГЛАВА ({_N}{_SUFFIX})", re.IGNORECASE)
ARTICLE_VARIANT = re.compile(rf"Статья ({_N}{_SUFFIX})(?: ?\.)?(?=\s|$)", re.IGNORECASE)
# Раздел и гриф: только прописные и только весь абзац — по наблюдённым формам корпуса.
SECTION = re.compile(r"(?:ПОДРАЗДЕЛ|РАЗДЕЛ) ([IVXLC]+|[1-9][0-9]{0,2})(?:\. \S.*)?")
APPENDIX = re.compile(rf"(?:ПРИЛОЖЕНИЕ|Приложение)(?: ([1-9][0-9]{{0,2}}(?:{_SUFFIX})?))?")
APPENDIX_ADDRESSED = re.compile(rf"(?:ПРИЛОЖЕНИЕ|Приложение) ([1-9][0-9]{{0,2}}(?:{_SUFFIX})?) к \S.{{0,200}}")
APPROVAL_STAMP = re.compile(r"(?:УТВЕРЖДЕН|СОГЛАСОВАН)[ОАЫ]?")
STRUCTURAL_LIKE = re.compile(
    r"(?:ГЛАВА|Глава|СТАТЬЯ|Статья|ПОДРАЗДЕЛ|Подраздел|РАЗДЕЛ|Раздел|ПРИЛОЖЕНИЕ|Приложение)"
    r" ?(?:[0-9]|[IVXLC]+\b)|§ ?[0-9]")
# Ключевое слово + арабский номер + пробел + строчное слово: по снимку нельзя отличить заголовок-заглушку
# от повествовательной фразы. Граница сохраняется, неопределённость помечается.
HEADING_OR_PROSE = re.compile(
    r"(?:ГЛАВА|Глава|СТАТЬЯ|Статья|ПОДРАЗДЕЛ|Подраздел|РАЗДЕЛ|Раздел|ПРИЛОЖЕНИЕ|Приложение)"
    r" ?[0-9]+ [а-яё]")
ACT_TITLE = re.compile(r"(?:ДЕКРЕТ|УКАЗ|ПОСТАНОВЛЕНИЕ|ЗАКОН|КОДЕКС|ПОЛОЖЕНИЕ|ИНСТРУКЦИЯ|ПРАВИЛА|"
                       r"РЕШЕНИЕ|ПРИКАЗ|РАСПОРЯЖЕНИЕ|ПОРЯДОК|РЕГЛАМЕНТ)\b")
# Набранная нумерация: точка в конце обязательна (исключает дроби и даты), без ведущих нулей.
NUM_INSERTED = re.compile(rf"([1-9][0-9]{{0,2}}{_SUFFIX})\.(?=\s|$)")
NUM_DOTTED = re.compile(rf"([1-9][0-9]{{0,2}}(?:\.[1-9][0-9]{{0,2}}){{1,4}}(?:{_SUFFIX})?)\.(?=\s|$)")
NUM_SINGLE = re.compile(r"([1-9][0-9]{0,2})\.(?=\s|$)")
OPEN_QUOTES = "\"«“„"
QUOTED_MARKER = re.compile(
    rf"[{OPEN_QUOTES}] ?(?:[0-9]{{1,3}}(?:\.[0-9]{{1,3}})*(?:-[0-9]{{1,2}})?[.)]"
    r"|(?:ГЛАВА|Глава|СТАТЬЯ|Статья|РАЗДЕЛ|Раздел|ПРИЛОЖЕНИЕ|Приложение)\b)")
AMENDMENT_INTRO = re.compile(r"(?:в следующей редакции|следующего содержания)\s*:$", re.IGNORECASE)
CLOSING_QUOTE = re.compile(r"[»\"”“][.;,]?$")
CONTENT_LABELS = frozenset({"text", "paragraph", "inline", "section_header", "title", "list_item"})
TYPED_NUMBERING_LABELS = frozenset({"text", "paragraph", "inline"})
_HEADING_LIKE = frozenset({"chapter", "chapter_variant", "section", "appendix"})

ItemKind = Literal["content", "opaque", "unsupported", "other_layer", "other_label"]
Effect = Literal["domain_boundary", "stop_only"]
Evidence = Literal["textual_marker", "heuristic_pattern", "snapshot_structure", "numbering_sequence",
                   "quoted_span"]
Confidence = Literal["confirmed", "heuristic", "unknown"]
QuotedContext = Literal["none", "suspected", "unknown"]

# Единственный источник машинных кодов этого модуля и их русских пояснений.
CODES: dict[str, str] = {
    "chapter": "изолированное «ГЛАВА N» (правило 1.2a)",
    "chapter_variant": "изолированное «ГЛАВА N-M»; правовой статус вставки не установлен",
    "article": "«Статья N.» или «Статья N» (правило 1.2a)",
    "article_variant": "«Статья N-M.» / «Статья N-M .» / «Статья N-M»; не тождественна статье N",
    "section": "«РАЗДЕЛ»/«ПОДРАЗДЕЛ» прописными с римским/арабским номером, весь абзац",
    "appendix": "«Приложение» [N] весь абзац; со строкой «к …» — эвристика",
    "approval_stamp": "гриф «УТВЕРЖДЕНО»/«СОГЛАСОВАНО» прописными, весь абзац; юридическая сила не установлена",
    "unrecognized_structural_marker": "похоже на главу/статью/раздел/приложение/§, но форма не поддержана",
    "act_title_candidate": "строка прописными с названием вида акта; не доказывает начало акта",
    "unrecognized_heading": "заголовок Docling без поддержанного маркера",
    "quoted_block_start": "первый абзац в кавычках после вводной формулы изменения",
    "quoted_block_end": "первый абзац после закрывающей кавычки цитируемого блока",
    "amendment_text_undelimited": "после вводной формулы изменения нет открывающей кавычки; граница текста неизвестна",
    "numbering_restart": "«1.» повторно в том же домене без распознанной границы",
    "quoted_marker": "абзац начинается с кавычки и маркера; это цитата, не нумерация документа",
    "opaque_block": "таблица, изображение или иной нетекстовый узел; номера внутри не используются",
    "unsupported_structure": "сложный InlineGroup; содержимое не разобрано",
    "non_body_layer": "элемент дерева body в слое, отличном от body",
    "other_label": "сноска, подпись, код или иная метка; номера не используются",
    "domain_boundary": "закрывает текущий домен нумерации и открывает следующий",
    "stop_only": "запрещает будущее присоединение через себя; домен не меняет",
    "textual_marker": "точное совпадение поддержанного текстового маркера",
    "heuristic_pattern": "эвристический шаблон; возможен ложный кандидат",
    "snapshot_structure": "тип узла снимка Docling, а не текст",
    "numbering_sequence": "вывод только из последовательности номеров",
    "quoted_span": "граница предполагаемого цитируемого блока",
    "confirmed": "текстовый маркер найден; правовой смысл не проверен",
    "heuristic": "кандидат по эвристике",
    "unknown": "основание или протяжённость не установлены",
    "none": "признаков цитируемого контекста нет",
    "suspected": "внутри предполагаемого цитируемого блока",
    "body_start": "начало потока body",
    "end_of_body": "конец потока body",
    "closing_quote": "блок закрыт абзацем с закрывающей кавычкой",
    "not_found": "закрывающая кавычка не найдена; всё после начала блока — quoted_context unknown",
    "single": "«N.» в тексте снимка",
    "dotted": "«N.M.», «N.M.K.» … в тексте снимка",
    "inserted": "вставной «N-M.» в тексте снимка; не открывает новый домен",
    "dotted_inserted": "«N.M-K.» в тексте снимка",
    "list_item_typed": "номер набран в тексте ListItem; соответствие автонумерации не проверено",
    "list_marker_computed": "номер есть только в вычисленном ListItem.marker Docling; не используется",
    "typed_in_snapshot_text": "номер присутствует в тексте снимка; соответствие оригиналу NOT_VERIFIED",
    "unverified_list_item": "номер списка Docling не проверен; исключён из последовательностей",
    "first": "первый номер верхнего уровня в домене",
    "next": "на единицу больше предыдущего номера верхнего уровня",
    "gap": "больше ожидаемого следующего номера",
    "repeat": "повтор предыдущего номера",
    "backward": "меньше предыдущего номера, но не 1",
    "inserted_after_base": "вставной номер после номера с той же основой",
    "inserted_unmatched": "вставной номер без предшествующей основы в домене",
    "sublevel": "многоуровневый номер; последовательность не сравнивается, родитель — по префиксу (1.2a.1-B)",
    "excluded_unverified": "исключён из последовательностей",
    "observed_continues_previous": "первый номер на 1 больше последнего номера прошлого сопоставимого домена; это наблюдение, не единый домен",
    "observed_restarts_at_1": "нумерация домена начинается с 1",
    "observed_other": "первый номер не продолжает прошлый домен и не равен 1",
    "after_quoted_block": "между этим и прошлым сопоставимым доменом есть цитируемый контекст",
    "not_applicable": "нет номеров верхнего уровня или сопоставимого прошлого домена",
    "quoted_context_suspected": "маркер внутри предполагаемого цитируемого блока; понижен до эвристики",
    "quoted_context_unknown": "закрытие цитируемого блока не найдено; принадлежность тексту документа неизвестна",
    "marker_form_unsupported": "форма маркера не поддержана политикой",
    "appendix_addressee_line": "приложение распознано по строке «… к …»; возможна фраза текста",
    "after_heading_not_boundary": "строка прописными/заголовок сразу после главы/раздела/приложения — вероятное название, не граница",
    "heading_or_prose_undetermined": "за номером следует строчное слово: заголовок-заглушка или фраза текста — не установлено; барьер сохранён",
}


@dataclass(frozen=True)
class StreamItem:
    """Элемент упорядоченного потока body. `cue` — только представление для распознавания."""

    order: int
    source_refs: tuple[str, ...]
    kind: ItemKind
    label: str
    cue: str = ""
    computed_marker: bool = False  # ListItem.enumerated: номер вычисляет Docling, не текст снимка


@dataclass(frozen=True)
class Boundary:
    id: str
    order: int
    source_refs: tuple[str, ...]
    category: str
    effect: Effect
    evidence: Evidence
    confidence: Confidence
    designation: str | None
    quoted_context: QuotedContext
    uncertainty: tuple[str, ...] = ()


@dataclass(frozen=True)
class NumberingMarker:
    id: str
    order: int
    source_refs: tuple[str, ...]
    form: str
    designation: str | None
    level: int | None
    domain_id: str
    sequence: str
    verification: str
    quoted_context: QuotedContext


@dataclass(frozen=True)
class QuotedSpan:
    id: str
    intro_order: int
    start_order: int
    start_refs: tuple[str, ...]
    end_order: int | None
    end_refs: tuple[str, ...]
    end_basis: Literal["closing_quote", "not_found"]


@dataclass
class NumberingDomain:
    id: str
    index: int
    start_order: int
    start_refs: tuple[str, ...]
    end_order: int
    end_refs: tuple[str, ...]
    opened_by: str | None
    closed_by: str | None
    start_basis: str
    end_basis: str
    confidence: Confidence
    quoted_context: QuotedContext
    marker_count: int = 0
    level1_count: int = 0
    first_level1: str | None = None
    last_level1: str | None = None
    last_marker_order: int | None = None
    last_marker_refs: tuple[str, ...] = ()
    continuity: str = "not_applicable"
    sequence_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Cue:
    category: str
    evidence: Evidence
    confidence: Confidence
    designation: str | None
    uncertainty: tuple[str, ...] = ()


def article_match(cue: str) -> re.Match[str] | None:
    """Правило 1.2a: составное обозначение («Статья 12.1», «Статья 12-1») не принимается за N."""
    match = ARTICLE.match(cue)
    if match and match.end() < len(cue) and not cue[match.end()].isspace():
        return None
    return match


def _structural(item: StreamItem, previous: _Cue | None) -> tuple[_Cue | None, bool]:
    """Категория доменной границы и признак названия сразу после главы/раздела/приложения."""
    cue = item.cue
    if cue[:1] in OPEN_QUOTES:
        return None, False
    found = _marker(cue)
    if found is not None:
        return found, False
    after_heading = previous is not None and previous.category in _HEADING_LIKE
    letters = [char for char in cue if char.isalpha()]
    if ACT_TITLE.match(cue) and len(cue) <= 200 and all(char.isupper() for char in letters):
        return (None, True) if after_heading else (_Cue("act_title_candidate", "heuristic_pattern", "heuristic",
                                                        None), False)
    if item.label in {"section_header", "title"}:
        return (None, True) if after_heading else (_Cue("unrecognized_heading", "snapshot_structure",
                                                        "heuristic", None), False)
    return None, False


def _marker(cue: str) -> _Cue | None:
    if match := CHAPTER.fullmatch(cue):
        return _Cue("chapter", "textual_marker", "confirmed", match.group(1))
    if match := CHAPTER_VARIANT.fullmatch(cue):
        return _Cue("chapter_variant", "textual_marker", "confirmed", match.group(1))
    if match := article_match(cue):
        return _Cue("article", "textual_marker", "confirmed", match.group(1))
    if match := ARTICLE_VARIANT.match(cue):
        return _Cue("article_variant", "textual_marker", "confirmed", match.group(1))
    if match := SECTION.fullmatch(cue):
        return _Cue("section", "textual_marker", "confirmed", match.group(1))
    if match := APPENDIX.fullmatch(cue):
        return _Cue("appendix", "textual_marker", "confirmed", match.group(1))
    if match := APPENDIX_ADDRESSED.fullmatch(cue):
        return _Cue("appendix", "heuristic_pattern", "heuristic", match.group(1), ("appendix_addressee_line",))
    if APPROVAL_STAMP.fullmatch(cue):
        return _Cue("approval_stamp", "textual_marker", "confirmed", None)
    if STRUCTURAL_LIKE.match(cue):
        return _Cue("unrecognized_structural_marker", "heuristic_pattern", "heuristic", None,
                    ("marker_form_unsupported",)
                    + (("heading_or_prose_undetermined",) if HEADING_OR_PROSE.match(cue) else ()))
    return None


def marker_category(cue: str) -> str | None:
    """Категория структурного маркера в начале текста, без учёта контекста потока."""
    found = _marker(cue)
    return found.category if found else None


def _numbering(item: StreamItem) -> tuple[str, str | None, int | None, str] | None:
    """Форма, обозначение, уровень и проверяемость номера в начале элемента."""
    cue = item.cue
    if cue[:1] in OPEN_QUOTES:
        return None
    typed = None
    if match := NUM_INSERTED.match(cue):
        typed = ("inserted", match.group(1), 1)
    elif match := NUM_DOTTED.match(cue):
        designation = match.group(1)
        form = "dotted_inserted" if "-" in designation else "dotted"
        typed = (form, designation, designation.split("-")[0].count(".") + 1)
    elif match := NUM_SINGLE.match(cue):
        typed = ("single", match.group(1), 1)
    if item.label == "list_item":
        if typed:
            return "list_item_typed", None, None, "unverified_list_item"
        return ("list_marker_computed", None, None, "unverified_list_item") if item.computed_marker else None
    if typed and item.label in TYPED_NUMBERING_LABELS:
        return (*typed, "typed_in_snapshot_text")
    return None


def _closes_quote(cue: str, opening: bool) -> bool:
    """Абзац закрывает цитату: оканчивается кавычкой и баланс кавычек это подтверждает.

    «» считаются по балансу; прямые и „“” — по чётности, так как их направление неоднозначно.
    """
    if not CLOSING_QUOTE.search(cue):
        return False
    odd = sum(cue.count(char) for char in "\"“”„") % 2 == 1
    balance = cue.count("«") - cue.count("»")
    if opening:
        return balance <= 0 if cue[:1] == "«" else not odd
    return odd or balance < 0


def _quoted_spans(stream: list[StreamItem], make_id: Callable[..., str]
                  ) -> tuple[list[QuotedSpan], dict[int, QuotedContext], set[int]]:
    """Цитируемые блоки открываются только вводной формулой + абзацем с кавычкой.

    Блок только добавляет границы и понижает уверенность; он не подавляет маркеры внутри.
    """
    spans: list[QuotedSpan] = []
    context: dict[int, QuotedContext] = {}
    undelimited: set[int] = set()
    index = 0
    while index < len(stream):
        item = stream[index]
        following = _next_content(stream, index + 1)
        if item.kind != "content" or not AMENDMENT_INTRO.search(item.cue) or following is None:
            index += 1
            continue
        start = stream[following]
        if following != index + 1 or start.cue[:1] not in OPEN_QUOTES:
            undelimited.add(following)
            index += 1
            continue
        end = None
        for position in range(following, len(stream)):
            candidate = stream[position]
            if candidate.kind == "content" and _closes_quote(candidate.cue, position == following):
                end = position
                break
        span_id = make_id("quoted-span", start.source_refs[0])
        if end is None:
            spans.append(QuotedSpan(span_id, item.order, start.order, start.source_refs, None, (), "not_found"))
            for position in range(following, len(stream)):
                context[position] = "unknown"
            break
        spans.append(QuotedSpan(span_id, item.order, start.order, start.source_refs,
                                stream[end].order, stream[end].source_refs, "closing_quote"))
        for position in range(following, end + 1):
            context[position] = "suspected"
        index = end + 1
    return spans, context, undelimited


def _next_content(stream: list[StreamItem], position: int) -> int | None:
    return next((index for index in range(position, len(stream)) if stream[index].kind == "content"), None)


def analyze(stream: list[StreamItem], make_id: Callable[..., str]) -> dict[str, Any]:
    """Построить границы, маркеры нумерации, цитируемые блоки и домены для одного потока."""
    spans, context, undelimited = _quoted_spans(stream, make_id)
    span_starts = {span.start_order for span in spans}
    span_after_end: set[int] = set()
    for span in spans:
        if span.end_order is not None:
            end = next(position for position, item in enumerate(stream) if item.order == span.end_order)
            after = _next_content(stream, end + 1)
            if after is not None:
                span_after_end.add(after)
    boundaries: list[Boundary] = []
    markers: list[NumberingMarker] = []
    domains: list[NumberingDomain] = []
    stats: Counter[str] = Counter()
    level1: list[int] = []
    issues: set[str] = set()
    previous_cue: _Cue | None = None

    def add_boundary(item: StreamItem, category: str, effect: Effect, evidence: Evidence,
                     confidence: Confidence, designation: str | None, quoted: QuotedContext,
                     uncertainty: tuple[str, ...] = ()) -> Boundary:
        boundary = Boundary(make_id("boundary", item.source_refs[0], category), item.order, item.source_refs,
                            category, effect, evidence, confidence, designation, quoted, uncertainty)
        boundaries.append(boundary)
        return boundary

    def open_domain(item: StreamItem, boundary: Boundary | None, quoted: QuotedContext) -> None:
        if domains:
            close_domain(boundary)
        domains.append(NumberingDomain(
            make_id("domain", item.source_refs[0]), len(domains), item.order, item.source_refs,
            item.order, item.source_refs, boundary.id if boundary else None, None,
            boundary.category if boundary else "body_start", "end_of_body",
            boundary.confidence if boundary else "confirmed", quoted))
        level1.clear()
        issues.clear()

    def close_domain(boundary: Boundary | None) -> None:
        domain = domains[-1]
        domain.closed_by = boundary.id if boundary else None
        domain.end_basis = boundary.category if boundary else "end_of_body"
        domain.sequence_issues = tuple(sorted(issues))

    for position, item in enumerate(stream):
        quoted = context.get(position, "none")
        if item.kind != "content":
            if not domains:
                open_domain(item, None, quoted)
            domain = domains[-1]
            category = {"opaque": "opaque_block", "unsupported": "unsupported_structure",
                        "other_layer": "non_body_layer", "other_label": "other_label"}[item.kind]
            add_boundary(item, category, "stop_only", "snapshot_structure", "confirmed", None, quoted)
            domain.end_order, domain.end_refs = item.order, item.source_refs
            # Название «сразу после» главы/раздела/приложения — только в непрерывном логическом потоке:
            # барьер разрывает смежность; домен при этом не меняется.
            previous_cue = None
            continue
        cue, title_after_heading = _structural(item, previous_cue)
        stats["after_heading_not_boundary"] += title_after_heading
        opener: Boundary | None = None
        # Одна доменная граница на позицию: структурная граница важнее вспомогательной quoted_block_end.
        # Где закрылась цитата, авторитетно сообщает quoted_spans.end_order/end_refs, а quoted_context
        # «none» у этой структурной границы показывает, что цитата закрылась до неё.
        if cue is not None:
            confidence, uncertainty = cue.confidence, cue.uncertainty
            if quoted != "none":
                confidence = "heuristic" if confidence == "confirmed" else confidence
                uncertainty = (*uncertainty, f"quoted_context_{quoted}")
            opener = add_boundary(item, cue.category, "domain_boundary", cue.evidence, confidence,
                                  cue.designation, quoted, uncertainty)
        elif item.order in span_starts:
            opener = add_boundary(item, "quoted_block_start", "domain_boundary", "quoted_span", "heuristic",
                                  None, quoted)
        elif position in span_after_end:
            opener = add_boundary(item, "quoted_block_end", "domain_boundary", "quoted_span", "heuristic",
                                  None, quoted)
        elif position in undelimited:
            opener = add_boundary(item, "amendment_text_undelimited", "domain_boundary", "heuristic_pattern",
                                  "unknown", None, quoted)
        if item.cue[:1] in OPEN_QUOTES and QUOTED_MARKER.match(item.cue):
            add_boundary(item, "quoted_marker", "stop_only", "textual_marker", "confirmed", None, quoted)
        # Абзац, открывающий домен цитаты/после цитаты, может нести собственный номер.
        number = _numbering(item)
        if (opener is None and number and number[0] == "single" and number[1] == "1" and level1):
            opener = add_boundary(item, "numbering_restart", "domain_boundary", "numbering_sequence", "unknown",
                                  "1", quoted)
        if opener is not None or not domains:
            open_domain(item, opener, quoted)
        domain = domains[-1]
        if quoted != "none" and domain.quoted_context == "none":
            domain.quoted_context = quoted
        domain.end_order, domain.end_refs = item.order, item.source_refs
        previous_cue = cue
        if number is None:
            continue
        form, designation, level, verification = number
        if verification != "typed_in_snapshot_text":
            sequence = "excluded_unverified"
        elif form == "single":
            value = int(designation)
            sequence = ("first" if not level1 else "next" if value == level1[-1] + 1 else "repeat"
                        if value == level1[-1] else "gap" if value > level1[-1] else "backward")
            level1.append(value)
            domain.level1_count += 1
            domain.first_level1 = domain.first_level1 or designation
            domain.last_level1 = designation
        elif form == "inserted":
            sequence = ("inserted_after_base" if level1 and int(designation.split("-")[0]) == level1[-1]
                        else "inserted_unmatched")
        else:
            sequence = "sublevel"
        if sequence in {"gap", "repeat", "backward", "inserted_unmatched"}:
            issues.add(sequence)
        markers.append(NumberingMarker(make_id("marker", item.source_refs[0]), item.order, item.source_refs,
                                       form, designation, level, domain.id, sequence, verification, quoted))
        domain.marker_count += 1
        if verification == "typed_in_snapshot_text":
            domain.last_marker_order, domain.last_marker_refs = item.order, item.source_refs
    if domains:
        close_domain(None)
    _continuity(domains)
    return {"boundaries": boundaries, "numbering_markers": markers, "quoted_spans": spans,
            "numbering_domains": domains, "stats": stats}


def _continuity(domains: list[NumberingDomain]) -> None:
    """Наблюдение о нумерации относительно прошлого сопоставимого домена; не слияние доменов."""
    comparable: NumberingDomain | None = None
    quoted_between = False
    for domain in domains:
        if domain.quoted_context != "none":
            quoted_between = True
            continue
        if domain.first_level1 is None:
            continue
        if comparable is None:
            domain.continuity = "observed_restarts_at_1" if domain.first_level1 == "1" else "not_applicable"
        elif quoted_between:
            domain.continuity = "after_quoted_block"
        elif int(domain.first_level1) == int(comparable.last_level1 or 0) + 1:
            domain.continuity = "observed_continues_previous"
        elif domain.first_level1 == "1":
            domain.continuity = "observed_restarts_at_1"
        else:
            domain.continuity = "observed_other"
        comparable, quoted_between = domain, False
