"""Источник LE-2: закреплённый RTF → UTF-8 TXT LibreOffice → точный фрагмент пунктов и карта строк.

Доказательством для эксперимента служит выбранный TXT, а не байты RTF. Реальные фразы-якоря,
маркеры структуры и контрольные значения задаются частной спецификацией в runtime и в Git
не попадают. Все проверки закрываются отказом: при неоднозначности фрагмент не создаётся.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

CONVERT_FILTER = "txt:Text (encoded):UTF8"
CONVERT_TIMEOUT_S = 120
BOM = "\ufeff"
POINT_START = re.compile(r"(\d{1,3})\. ")
# Служебная строка абзаца RTF перед текстом: только сброс и нейтральное оформление.
RTF_PREFIX_WORD = re.compile(
    rb"\\(?:pard|plain|s\d+|f\d+|fs\d+|q[ljcr]|fi-?\d+|li-?\d+|ri-?\d+|sb\d+|sa\d+"
    rb"|b0|i0|nosupersub|ul0|strike0|itap0)"
)
SPEC_KEYS = {"source_id", "rtf_sha256", "rtf_size", "rtf_codepage", "path_markers", "instrument_marker",
             "closing_marker", "start_anchor", "end_anchor", "points", "reference_values", "structural_path"}


class SourceError(Exception):
    """Источник не совпал с закреплённым или фрагмент нельзя выделить однозначно."""


@dataclass(frozen=True)
class Line:
    """Строка TXT внутри пункта; смещения — символы Unicode фрагмента, [start, end) без '\\n'."""

    id: str
    point: str
    line_no: int
    start: int
    end: int
    abzac: int | None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_spec(data: bytes) -> dict[str, Any]:
    spec = json.loads(data.decode("utf-8"))
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS:
        raise SourceError("Спецификация источника не совпадает с ожидаемым набором ключей")
    if len(spec["points"]) < 1 or not all(isinstance(p, str) and p.isdigit() for p in spec["points"]):
        raise SourceError("В спецификации нужен непустой список номеров пунктов")
    return spec


def convert(soffice: str, rtf: Path, work: Path) -> bytes:
    """Одно преобразование в новом профиле LibreOffice; сеть должна быть отключена снаружи (unshare -rn)."""
    with tempfile.TemporaryDirectory(prefix=".soffice-", dir=work) as temporary:
        profile, out = Path(temporary, "profile"), Path(temporary, "out")
        profile.mkdir(mode=0o700)
        out.mkdir(mode=0o700)
        environment = {key: value for key, value in os.environ.items()
                       if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
        try:
            subprocess.run([soffice, f"-env:UserInstallation={profile.as_uri()}", "--headless", "--norestore",
                            "--nolockcheck", "--nodefault", "--nologo", "--convert-to", CONVERT_FILTER,
                            "--outdir", str(out), str(rtf)],
                           capture_output=True, timeout=CONVERT_TIMEOUT_S, check=True, env=environment)
        except (OSError, subprocess.SubprocessError):
            raise SourceError("LibreOffice не выполнил преобразование") from None
        produced = list(out.iterdir())
        if len(produced) != 1 or produced[0].suffix != ".txt":
            raise SourceError("LibreOffice не создал ровно один TXT")
        return produced[0].read_bytes()


def _unique(text: str, needle: str, what: str) -> int:
    first = text.find(needle)
    if first < 0 or text.find(needle, first + 1) >= 0:
        raise SourceError(f"{what}: нужно ровно одно вхождение")
    return first


def _full_line(text: str, line: str, what: str) -> int:
    found = [m.start() for m in re.finditer(rf"(?m)^{re.escape(line)}$", text)]
    if len(found) != 1:
        raise SourceError(f"{what}: нужна ровно одна строка, найдено {len(found)}")
    return found[0]


def locate(text: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Найти границы фрагмента и подтвердить, что он лежит в нужном акте и главе."""
    markers = [_full_line(text, marker, "Маркер структуры") for marker in spec["path_markers"]]
    if markers != sorted(markers):
        raise SourceError("Маркеры структуры идут не в ожидаемом порядке")
    instrument = markers[spec["path_markers"].index(spec["instrument_marker"])]
    start = _unique(text, spec["start_anchor"], "Начальный якорь")
    end = _unique(text, spec["end_anchor"], "Конечный якорь")
    closing = _full_line(text, spec["closing_marker"], "Маркер конца главы")
    if not markers[-1] < start < end < closing:
        raise SourceError("Якоря не лежат между последним маркером структуры и маркером конца главы")
    if (start and text[start - 1] != "\n") or text[end - 1] != "\n":
        raise SourceError("Якоря должны начинать строку")
    points = spec["points"]
    excerpt = text[start:end]
    labels = [m[1] for m in re.finditer(r"(?m)^(\d{1,3})\. ", excerpt)]
    if labels != points or POINT_START.match(excerpt)[1] != points[0]:
        raise SourceError(f"Начала пунктов во фрагменте {labels} не совпадают с {points}")
    if POINT_START.match(text, end)[1] != str(int(points[-1]) + 1):
        raise SourceError("Конечный якорь не является следующим по номеру пунктом")
    scope = text[instrument:]
    in_instrument = {label: len(re.findall(rf"(?m)^{label}\. ", scope)) for label in [*points, str(int(points[-1]) + 1)]}
    if any(count != 1 for count in in_instrument.values()):
        raise SourceError(f"Номера пунктов в акте неуникальны: {in_instrument}")
    elsewhere = {label: len(re.findall(rf"(?m)^{label}\. ", text[:instrument])) for label in points}
    return {"start": start, "end": end, "marker_offsets": dict(zip(spec["path_markers"], markers)),
            "closing_marker_offset": closing, "point_line_starts_in_instrument": in_instrument,
            "same_numbers_before_instrument": elsewhere}


def _is_enumeration(lines: list[str]) -> bool:
    """Пункт-перечень «вводная: …; …; ….»: только в нём абзац N однозначно равен строке N."""
    return (len(lines) >= 2 and lines[0].endswith(":") and lines[-1].endswith(".")
            and all(line.endswith(";") for line in lines[1:-1]))


def line_units(excerpt: str, points: list[str]) -> list[Line]:
    """Карта строк по пунктам; отделена от проверки цитат. Пустые строки и '\\r' не допускаются."""
    if "\r" in excerpt or not excerpt.endswith("\n"):
        raise SourceError("Фрагмент должен состоять из строк с '\\n' и заканчиваться '\\n'")
    raw, offset = [], 0
    for line in excerpt[:-1].split("\n"):
        if not line.strip():
            raise SourceError("Пустая строка внутри фрагмента: границы абзацев неоднозначны")
        raw.append((offset, line))
        offset += len(line) + 1
    groups: list[list[tuple[int, str]]] = []
    for start, line in raw:
        match = POINT_START.match(line)
        if match:
            groups.append([])
        if not groups:
            raise SourceError("Фрагмент начинается не с пункта")
        groups[-1].append((start, line))
    if [POINT_START.match(group[0][1])[1] for group in groups] != points:
        raise SourceError("Состав пунктов фрагмента не совпадает со спецификацией")
    units = []
    for point, group in zip(points, groups):
        enumeration = _is_enumeration([line for _, line in group])
        for number, (start, line) in enumerate(group, 1):
            units.append(Line(f"{point}:L{number}", point, number, start, start + len(line),
                              number if enumeration else None))
    return units


def rtf_crosscheck(rtf: bytes, spec: dict[str, Any], excerpt: str) -> dict[str, Any]:
    """Независимая узкая сверка: абзацы RTF между якорями (cp1251, \\pard…\\par) равны строкам TXT.

    Не общий разбор RTF: любой неожиданный управляющий элемент или группа в тексте — отказ.
    """
    codepage = spec["rtf_codepage"]
    start_anchor = spec["start_anchor"].encode(codepage)
    end_anchor = spec["end_anchor"].encode(codepage)
    start = rtf.find(start_anchor)
    end = rtf.find(end_anchor)
    if start < 0 or rtf.find(start_anchor, start + 1) >= 0 or end < 0 or rtf.find(end_anchor, end + 1) >= 0:
        raise SourceError("RTF: якоря должны встречаться ровно один раз")
    pieces = rtf[start:end].split(b"\\par\n")
    texts, prefixes = [], set()
    for index, piece in enumerate(pieces):
        if index:
            if not piece.startswith(b"\\pard") or b"\n" not in piece:
                raise SourceError("RTF: абзац без ожидаемой служебной строки \\pard")
            prefix, piece = piece.split(b"\n", 1)
            if RTF_PREFIX_WORD.sub(b"", prefix).strip():
                raise SourceError("RTF: неожиданное оформление абзаца (возможен скрытый или особый текст)")
            prefixes.add(prefix.decode("ascii"))
        if any(char in piece for char in (b"\\", b"{", b"}", b"\n", b"\r")):
            raise SourceError("RTF: в тексте абзаца есть управляющие элементы; узкая сверка неприменима")
        texts.append(piece.decode(codepage))
    if texts[-1] != "":
        raise SourceError("RTF: после последнего абзаца перед конечным якорем есть текст")
    rendered = "\n".join(texts[:-1]) + "\n"
    return {"paragraphs": len(texts) - 1, "byte_range": [start, end], "distinct_prefixes": sorted(prefixes),
            "equals_txt_excerpt": rendered == excerpt}


def build_source_map(rtf: bytes, txt_bytes: bytes, spec: dict[str, Any], converter: str) -> tuple[str, dict[str, Any]]:
    """Проверить TXT, выделить фрагмент и собрать карту источника; вернуть фрагмент и карту."""
    if len(rtf) != spec["rtf_size"] or sha256(rtf) != spec["rtf_sha256"]:
        raise SourceError("RTF не совпадает с закреплёнными размером и SHA-256; другая редакция не подставляется")
    try:
        text = txt_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise SourceError("Результат преобразования не является UTF-8") from None
    if "\r" in text:
        raise SourceError("TXT содержит '\\r'")
    bounds = locate(text, spec)
    excerpt = text[bounds["start"]:bounds["end"]]
    excerpt_bytes = excerpt.encode("utf-8")
    observed = {"txt_bytes": len(txt_bytes), "txt_chars": len(text), "start": bounds["start"],
                "end": bounds["end"], "excerpt_chars": len(excerpt), "excerpt_bytes": len(excerpt_bytes)}
    mismatched = sorted(key for key, value in spec["reference_values"].items() if observed.get(key) != value)
    if mismatched:
        raise SourceError(f"Наблюдаемые значения расходятся с контрольными: {mismatched}")
    units = line_units(excerpt, spec["points"])
    crosscheck = rtf_crosscheck(rtf, spec, excerpt)
    if not crosscheck["equals_txt_excerpt"] or crosscheck["paragraphs"] != len(units):
        raise SourceError("Сверка с абзацами RTF не прошла")
    return excerpt, {
        "source_id": spec["source_id"],
        "structural_path": spec["structural_path"],
        "rtf": {"size": len(rtf), "sha256": sha256(rtf), "codepage": spec["rtf_codepage"]},
        "converter": {"name": converter, "filter": CONVERT_FILTER, "isolated_profile": True},
        "txt": {"bytes": len(txt_bytes), "chars": len(text), "sha256": sha256(txt_bytes), "bom": text.startswith(BOM)},
        "offsets": "символы Unicode TXT, декодированного как utf-8 (BOM = символ 0); [start, end)",
        "excerpt": {"start": bounds["start"], "end": bounds["end"], "chars": len(excerpt),
                    "bytes": len(excerpt_bytes), "sha256": sha256(excerpt_bytes),
                    "ends_with_newline_before_next_point": True},
        "bounds": bounds,
        "reference_values_match": True,
        "units": [unit.__dict__ for unit in units],
        "unit_policy": ("строка TXT = абзац RTF (\\par); номер абзаца пункта задан только для пункта-перечня "
                        "(вводная «:», элементы «;», последний «.»); для остальных пунктов abzac = null"),
        "rtf_crosscheck": crosscheck,
        "caveats": [
            "TXT — производное представление LibreOffice; оформление, нумерация и сноски RTF не сохраняются",
            "смещения включают BOM: utf-8-sig сдвинул бы их на 1",
            "соответствие RTF проверено только для текста абзацев выбранного участка, не байт в байт",
            "юридическая действительность и актуальность редакции не проверяются",
        ],
    }
