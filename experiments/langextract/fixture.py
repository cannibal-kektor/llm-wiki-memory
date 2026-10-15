"""Детерминированная синтетическая фикстура LE-1 и её эталонная разметка для оценки.

Текст полностью вымышлен и не является юридическим правилом. Эталонные номера и
диапазоны абзацев используются только для подсчёта метрик эксперимента; модели они
не передаются и не являются восстановленной правовой иерархией.
"""

from dataclasses import dataclass
import hashlib

FIXTURE_ID = "synthetic-warehouse-le1"
TITLE = "ПРАВИЛА ПОЛЬЗОВАНИЯ УСЛОВНЫМ СКЛАДОМ «АЛЬФА» (СИНТЕТИЧЕСКИЙ ПРИМЕР)"

# (заголовок главы, ((номер, текст положения без номера), ...))
CHAPTERS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Глава 1. Плата за хранение", (
        ("1", "Арендатор обязан вносить плату за хранение не позднее десятого числа каждого месяца."),
        ("2", "Арендатор вправе вносить плату за несколько месяцев вперёд при условии письменного "
              "уведомления оператора склада."),
    )),
    ("Глава 2. Доступ на склад", (
        ("3", "Доступ на склад предоставляется ежедневно с 8 до 20 часов."),
        ("4", "Запрещается хранить на складе легковоспламеняющиеся вещества."),
    )),
    ("Глава 3. Особые случаи", (
        ("5", "Требование пункта 1 не применяется к арендаторам, заключившим договор хранения на срок "
              "менее одного месяца; такие арендаторы вносят плату единовременно при заключении договора."),
        ("6", "Арендатор обязан уведомить оператора склада о повреждении упаковки груза в течение одних суток."),
        ("7", "Арендатор обязан уведомить оператора склада о прекращении договора не позднее чем за пять дней."),
    )),
)

# Ожидаемая явная связь для оценки LE-1B: положение-исключение → затрагиваемое положение.
EXPECTED_EXCEPTION_SOURCE = "5"
EXPECTED_EXCEPTION_TARGET = "1"
PARAGRAPH_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Provision:
    label: str
    paragraph_start: int
    paragraph_end: int
    body_start: int


def build_text() -> str:
    """Собрать текст: абзацы через пустую строку, «N. » перед положением, завершающий перевод строки."""
    paragraphs = [TITLE]
    for heading, provisions in CHAPTERS:
        paragraphs.append(heading)
        paragraphs.extend(f"{label}. {body}" for label, body in provisions)
    return PARAGRAPH_SEPARATOR.join(paragraphs) + "\n"


FIXTURE_TEXT = build_text()
FIXTURE_BYTES = FIXTURE_TEXT.encode("utf-8")
FIXTURE_SHA256 = hashlib.sha256(FIXTURE_BYTES).hexdigest()


def provisions(text: str = FIXTURE_TEXT) -> tuple[Provision, ...]:
    """Эталонные диапазоны абзацев положений в точном тексте фикстуры (символы Unicode, [start, end))."""
    result = []
    for _, items in CHAPTERS:
        for label, body in items:
            paragraph = f"{label}. {body}"
            start = text.find(paragraph)
            if start < 0 or text.find(paragraph, start + 1) >= 0:
                raise ValueError("Абзац эталонного положения отсутствует или неоднозначен")
            result.append(Provision(label, start, start + len(paragraph), start + len(label) + 2))
    return tuple(result)
