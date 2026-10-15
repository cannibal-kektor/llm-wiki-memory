"""CLI эксперимента LE-2: LangExtract 1.7.1 на точном фрагменте реального документа.

prepare — офлайн: закреплённый RTF → TXT LibreOffice → фрагмент пунктов + карта источника.
probe   — только GET к серверу модели; из /slots сохраняются лишь служебные поля.
run     — одна задача (passages = LE-2A или relations = LE-2B) с промптами и примерами LE-1 без
          изменений, одним вызовом модели, журналом POST до отправки и частным отчётом.
Повторного вызова, параллелизма, чанкинга и публикации нет; всё — UNVERIFIED / REVIEW_REQUIRED.

LE-3A (только офлайн, сетевые обращения процесса блокируются, модель не вызывается):
evidence-prepare — явный выбор строк → точные доказательства и ContextPackage;
claim-request    — промпт будущего шага Claim Construction из пакета, только в частный файл;
claim-check      — строгая проверка ответа модельной формы; смысл — NOT_VERIFIED.

LE-3B.1: claim-run — один вызов Claim Construction по заново построенному и сверенному запросу.
По умолчанию без сети: --replay (частный файл обмена) или --preflight (проверка условий без отправки).
--live отправляет не более одного POST и требует отдельного свежего разрешения владельца.
"""

import argparse
from collections import Counter
import dataclasses
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener
import warnings

from experiments.langextract.__main__ import (
    MAX_DIAGNOSTICS, NetworkGuard, Refused, _Capture, _clean_environment, _endpoint_target, _questions, _run_task,
)
from llm_wiki.ingest import resolve_libreoffice
from llm_wiki.llm import LLMConfig
from llm_wiki.structure import StructureError, _output_directory, _write_new
from llm_wiki.validation import parse_json

from . import claim_run, claims, evidence, source

SCHEMA = "llm-wiki-memory/langextract-real-experiment/v1"
LE1_SCHEMA = "llm-wiki-memory/langextract-experiment/v1"
MAX_INPUT_BYTES = 65_536
MAX_CHAR_BUFFER = 4000
MAX_TOKENS = 8192
PROBE_TIMEOUT_S = 10
SLOT_FIELDS = ("id", "id_task", "is_processing", "n_ctx")
TASKS = {"passages": ("LE-2A", 0), "relations": ("LE-2B", 1)}  # индекс — соответствующий вызов в отчёте LE-1


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _directory(path: str, names: tuple[str, ...]) -> Path:
    try:
        return _output_directory(Path(path), names)
    except StructureError as error:
        raise Refused(str(error)) from None


def _private_file(path: str, limit: int = 4_000_000) -> bytes:
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size > limit:
        raise Refused(f"Файл не найден или слишком велик: {candidate.name}")
    return candidate.read_bytes()


# --- prepare ---------------------------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> int:
    spec = source.load_spec(_private_file(args.spec))
    rtf = _private_file(args.rtf, 50_000_000)
    names = tuple(f"{spec['source_id']}.{suffix}" for suffix in ("full.txt", "excerpt.txt", "source-map.json"))
    directory = _directory(args.output_dir, names)
    try:
        soffice, version = resolve_libreoffice(Path(args.libreoffice) if args.libreoffice else None)
    except ValueError as error:
        raise Refused(str(error)) from None
    try:
        if len(rtf) != spec["rtf_size"] or source.sha256(rtf) != spec["rtf_sha256"]:
            raise source.SourceError("RTF не совпадает с закреплёнными размером и SHA-256")
        txt = source.convert(soffice, Path(args.rtf).resolve(), directory)
        excerpt, source_map = source.build_source_map(rtf, txt, spec, version)
    except source.SourceError as error:
        raise Refused(str(error)) from None
    source_map["created_utc"] = _utc()
    source_map["spec_sha256"] = source.sha256(_private_file(args.spec))
    for name, payload in zip(names, (txt, excerpt.encode("utf-8"), _json_bytes(source_map))):
        _write_new(directory, name, payload)
    print(f"Источник: RTF SHA-256 {source_map['rtf']['sha256']}; {version}")
    print(f"TXT: {source_map['txt']['bytes']} байт, {source_map['txt']['chars']} символов, "
          f"SHA-256 {source_map['txt']['sha256']}, BOM={source_map['txt']['bom']}")
    excerpt_info = source_map["excerpt"]
    print(f"Фрагмент: [{excerpt_info['start']}:{excerpt_info['end']}) {excerpt_info['chars']} символов, "
          f"{excerpt_info['bytes']} байт, SHA-256 {excerpt_info['sha256']}")
    print(f"Строк: {len(source_map['units'])}; сверка RTF: {source_map['rtf_crosscheck']['paragraphs']} абзацев, "
          f"совпадение={source_map['rtf_crosscheck']['equals_txt_excerpt']}; каталог (частный): {directory}")
    return 0


# --- probe -----------------------------------------------------------------------------------------

def probe(args: argparse.Namespace) -> int:
    config = LLMConfig(args.endpoint, "probe")
    try:
        allowed = _endpoint_target(config)
    except ValueError as error:
        raise Refused(str(error)) from None
    name = f"probe.{args.label}.{_utc()}.{time.monotonic_ns() % 1_000_000:06d}.json"
    directory = _directory(args.output_dir, (name,))
    guard = NetworkGuard(allowed)
    sys.addaudithook(guard)
    base = args.endpoint.rstrip("/").removesuffix("/v1")
    opener = build_opener(ProxyHandler({}))
    record: dict[str, Any] = {"label": args.label, "utc": datetime.now(timezone.utc).isoformat()}
    for path in ("/health", "/v1/models", "/slots", "/metrics"):
        started = time.monotonic()
        entry: dict[str, Any] = {}
        try:
            with opener.open(base + path, timeout=PROBE_TIMEOUT_S) as response:
                entry["status"], body = response.status, response.read(4_000_000)
            parsed = json.loads(body) if path != "/metrics" else None
            if path == "/health":
                entry["body"] = parsed
            elif path == "/v1/models":
                entry["ids"] = [model.get("id") for model in parsed.get("data", [])]
            elif path == "/slots":
                # Промпты и сгенерированный текст других клиентов не сохраняются и не выводятся.
                entry["slots"] = [{**{key: slot.get(key) for key in SLOT_FIELDS},
                                   "n_decoded": (slot.get("next_token") or {}).get("n_decoded")
                                   if isinstance(slot.get("next_token"), dict) else None,
                                   "n_predict": (slot.get("params") or {}).get("n_predict")} for slot in parsed]
        except HTTPError as error:
            entry["status"] = error.code
            error.close()
        except (URLError, OSError, ValueError, AttributeError) as error:
            entry["status"] = f"error:{type(error).__name__}"
        entry["seconds"] = round(time.monotonic() - started, 3)
        record[path] = entry
    _write_new(directory, name, _json_bytes(record))
    slots = record["/slots"].get("slots") or []
    print(f"health={record['/health']['status']} models={len(record['/v1/models'].get('ids') or [])} "
          f"slots_busy={[slot['is_processing'] for slot in slots]} metrics={record['/metrics']['status']}; {name}")
    return 0


# --- run -------------------------------------------------------------------------------------------

def _ledger_count(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        count += int(entry["count"]) if entry["kind"] == "baseline" else 1 if entry["kind"] == "post" else 0
    return count


def _ledgered(inner: Any, path: Path, cap: int, task: str) -> Any:
    """Запись в журнал POST до отправки: тайм-аут или обрыв тоже расходуют бюджет."""
    from experiments.langextract.model import TransportError

    def send(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        if _ledger_count(path) + 1 > cap:
            raise TransportError("budget_exceeded")
        line = json.dumps({"kind": "post", "utc": datetime.now(timezone.utc).isoformat(), "task": task,
                           "prompt_sha256": source.sha256(payload["messages"][-1]["content"].encode("utf-8"))})
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(descriptor, (line + "\n").encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return inner(url, payload, timeout)

    return send


def _gate(failure: Any, raw: list[dict[str, Any]], items: list[dict[str, Any]], accounting: dict[str, Any]) -> dict[str, Any]:
    """Условия перехода к LE-2B: ответ принят, разобран строго, без потерь и с точной привязкой каждого элемента."""
    reasons = []
    if failure:
        reasons.append(f"FAILURE:{failure['kind']}")
    if any(call["strict_parse"] != "OK" or call.get("error") or call.get("issues") for call in raw):
        reasons.append("RAW_NOT_STRICT")
    if accounting.get("absent_from_final") or accounting.get("not_in_raw"):
        reasons.append("RAW_FINAL_MISMATCH")
    if not items or any(item["grounding"] != "VERIFIED_EXACT_SPAN" for item in items):
        reasons.append("NOT_ALL_VERIFIED")
    if any(item["duplicate_of"] is not None for item in items):
        reasons.append("DUPLICATES")
    return {"status": "FAIL" if reasons else "PASS", "reasons": reasons}


def _load_inputs(args: argparse.Namespace) -> tuple[str, dict[str, Any], list[source.Line], int]:
    data = _private_file(args.input, MAX_INPUT_BYTES)
    source_map = json.loads(_private_file(args.source_map))
    if source.sha256(data) != source_map["excerpt"]["sha256"]:
        raise Refused("Вход не совпадает с фрагментом из карты источника")
    excerpt = data.decode("utf-8")
    points = list(dict.fromkeys(unit["point"] for unit in source_map["units"]))
    try:
        units = source.line_units(excerpt, points)
    except source.SourceError as error:
        raise Refused(str(error)) from None
    if [unit.__dict__ for unit in units] != source_map["units"]:
        raise Refused("Карта строк не воспроизводится из фрагмента")
    return excerpt, source_map, units, source_map["excerpt"]["start"]


def run(args: argparse.Namespace) -> int:
    excerpt, source_map, units, excerpt_offset = _load_inputs(args)
    task_name, le1_index = TASKS[args.task]
    checklist_bytes = _private_file(args.checklist) if args.checklist else None
    checklist = json.loads(checklist_bytes) if checklist_bytes else None
    le1_prompt_prefix = None
    if args.le1_report:
        le1 = json.loads(_private_file(args.le1_report))
        if le1.get("schema") != LE1_SCHEMA:
            raise Refused("--le1-report не является отчётом LE-1")
        prompt = le1["calls"][le1_index]["prompt"]
        le1_prompt_prefix = prompt[:prompt.rindex("\nQ: ")]

    passages_gate = None
    if args.task == "relations" and not args.replay:
        previous = json.loads(_private_file(args.passages_report))
        if (previous.get("schema") != SCHEMA or previous["task"] != "passages"
                or previous["configuration"]["mode"] != "http"
                or previous["source"]["excerpt_sha256"] != source_map["excerpt"]["sha256"]):
            raise Refused("--passages-report должен быть live-отчётом LE-2A (не replay) для того же фрагмента")
        passages_gate = previous["gate"]
        if passages_gate["status"] != "PASS":
            raise Refused(f"LE-2A не прошёл проверку ({passages_gate['reasons']}); LE-2B не запускается")

    replay = None
    ledger = None
    if args.replay:
        replay = json.loads(_private_file(args.replay))
        if replay.get("schema") != SCHEMA or replay.get("task") != args.task:
            raise Refused("Файл --replay не является отчётом LE-2 той же задачи")
        if replay["source"]["excerpt_sha256"] != source_map["excerpt"]["sha256"]:
            raise Refused("Отчёт --replay записан для другого фрагмента")
        recorded = replay["configuration"]
        config = LLMConfig(recorded["endpoint"], recorded["model"], timeout=recorded["timeout_s"],
                           response_format="json_object")
        allowed = None
    else:
        if not (args.endpoint and args.model and args.post_ledger):
            raise Refused("Для реального вызова нужны --endpoint, --model и --post-ledger (или --replay)")
        config = LLMConfig(args.endpoint, args.model, api_key=os.environ.get("LLM_WIKI_API_KEY"),
                           timeout=args.timeout, response_format="json_object")
        try:
            allowed = _endpoint_target(config)
        except ValueError as error:
            raise Refused(str(error)) from None
        ledger = Path(args.post_ledger)
        if not ledger.is_file() or ledger.stat().st_mode & 0o077:
            raise Refused("Журнал POST должен существовать (с базовой строкой) и иметь права 0600")
        if _ledger_count(ledger) + 1 > args.post_cap:
            raise Refused(f"Бюджет POST исчерпан: {_ledger_count(ledger)} из {args.post_cap}")

    created = _utc()
    report_name = f"{source_map['source_id']}.{args.task}.{'replay' if replay else 'http'}.{created}.le2-report.json"
    directory = _directory(args.output_dir, (report_name,))
    removed = _clean_environment()
    guard = NetworkGuard(allowed)
    sys.addaudithook(guard)
    capture = _Capture()
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.WARNING)

    import langextract as lx
    from langextract import chunking
    from langextract.core import tokenizer

    from experiments.langextract import grounding
    from experiments.langextract.model import GuardedChatModel, http_transport, replay_transport, to_dict
    from experiments.langextract.task import PASSAGES, RELATIONS

    from . import references

    regex_tokenizer = tokenizer.RegexTokenizer()
    plan = list(chunking.ChunkIterator(text=regex_tokenizer.tokenize(excerpt), max_char_buffer=MAX_CHAR_BUFFER,
                                       tokenizer_impl=regex_tokenizer))
    chunk = plan[0] if len(plan) == 1 else None
    if (chunk is None or chunk.char_interval.start_pos != 0
            or excerpt[chunk.char_interval.end_pos:].strip() or chunk.chunk_text != excerpt[:chunk.char_interval.end_pos]):
        raise Refused("Фрагмент не помещается в один чанк; дизайн без чанкинга не меняется молча")
    # Токенизатор LangExtract не включает завершающий '\n' в чанк; вне чанка допустимы только пробельные символы.
    chunk_interval = [0, chunk.char_interval.end_pos]

    # Промпт и примеры LE-1 без изменений; меняется только метка задачи в журнале вызовов.
    spec = dataclasses.replace(PASSAGES if args.task == "passages" else RELATIONS, name=task_name)
    transport = (replay_transport(replay["calls"]) if replay
                 else _ledgered(http_transport(config), ledger, args.post_cap, task_name))
    model = GuardedChatModel(config, transport, max_calls=1, max_tokens=MAX_TOKENS, temperature=None)
    options = argparse.Namespace(alignment="library-default", max_char_buffer=MAX_CHAR_BUFFER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        document, failure = _run_task(lx, spec, excerpt, model, options, 1)

    raw = [{"call": call.index, **(grounding.raw_items(call.content, spec.classes) if call.content is not None
                                   else {"strict_parse": "NO_CONTENT", "items": []})} for call in model.calls]
    prompt_checks = [{
        "question_equals_chunk": _questions(call.prompt) == chunk.chunk_text,
        "prefix_equals_le1": (call.prompt[:call.prompt.rindex("\nQ: ")] == le1_prompt_prefix
                              if le1_prompt_prefix is not None and "\nQ: " in call.prompt else None),
        "fenced": "```" in call.prompt,
    } for call in model.calls]
    items: list[dict[str, Any]] = []
    task: dict[str, Any] = {"status": "FAILED" if failure else "COMPLETED", "raw_per_call": raw,
                            "prompt_checks": prompt_checks}
    if document is not None:
        items = grounding.ground(excerpt, source_map["excerpt"]["sha256"], document.extractions or [],
                                 "passage" if args.task == "passages" else "relation")
        for item in items:  # ID строится функцией LE-1; префикс меняется, чтобы не путать с LE-1.
            item["id"] = item["id"] and "le2" + item["id"][3:]
        references.attach_units(items, excerpt, units, excerpt_offset)
        task.update(
            extractions=items,
            grounding_counts=dict(Counter(item["grounding"] for item in items)),
            library_alignment_counts=dict(Counter(str(item["library_alignment_status"]) for item in items)),
            attribution_counts=dict(Counter(item["source_attribution"] for item in items)),
            duplicates=[item["position"] for item in items if item["duplicate_of"] is not None],
            overlaps=grounding.overlaps(items),
            same_span_different_class=grounding.same_span_different_class(items),
            raw_vs_final=grounding.raw_vs_final(raw, items),
            non_verbatim_attributes=[item["position"] for item in items if item["non_verbatim_attributes"]],
        )
    # Проверка перехода относится к LE-2A; в LE-2B одно доказательство с двумя целями допустимо.
    gate = (_gate(failure, raw, items, task.get("raw_vs_final", {})) if args.task == "passages"
            else {"status": "NOT_APPLICABLE", "reasons": []})
    evaluation: dict[str, Any] = {}
    if args.task == "relations" and document is not None:
        task["resolved"] = [references.resolve(item, units) for item in items]
    if checklist and document is not None:
        evaluation = (references.evaluate_passages(checklist, items, excerpt, units) if args.task == "passages"
                      else references.evaluate_relations(checklist, task["resolved"], units))

    report = {
        "schema": SCHEMA, "task": args.task, "task_name": task_name, "created_utc": created,
        "outcome": "PIPELINE_FAILED" if failure else "PIPELINE_COMPLETED", "failure": failure, "gate": gate,
        "legal_status": "UNVERIFIED", "review_status": "REVIEW_REQUIRED",
        "versions": {"python": platform.python_version(),
                     **{name: importlib.metadata.version(name) for name in ("langextract", "openai")}},
        "source": {"source_id": source_map["source_id"], "structural_path": source_map["structural_path"],
                   "rtf_sha256": source_map["rtf"]["sha256"], "txt_sha256": source_map["txt"]["sha256"],
                   "excerpt_sha256": source_map["excerpt"]["sha256"],
                   "excerpt_full_txt_range": [source_map["excerpt"]["start"], source_map["excerpt"]["end"]],
                   "converter": source_map["converter"], "units": source_map["units"]},
        "checklist_sha256": source.sha256(checklist_bytes) if checklist_bytes else None,
        "passages_gate_used": passages_gate,
        "configuration": {
            "mode": "replay" if replay else "http", "replayed_from": Path(args.replay).name if replay else None,
            "endpoint": config.endpoint, "model": config.model, "provider": "GuardedChatModel (LE-1)",
            "response_format": "json_object", "timeout_s": config.timeout, "max_tokens": MAX_TOKENS,
            "temperature": "not sent (server default)", "max_calls": 1, "post_cap": None if replay else args.post_cap,
            "max_char_buffer": MAX_CHAR_BUFFER, "passes": 1, "batch_length": 1, "max_workers": 1,
            "alignment": "library-default", "prompt": f"LE-1 {'PASSAGES' if args.task == 'passages' else 'RELATIONS'}",
            "removed_environment": removed,
        },
        "chunks": {"count": 1, "interval": chunk_interval, "excluded_tail": excerpt[chunk_interval[1]:]},
        "calls": [to_dict(call) for call in model.calls],
        "result": task, "evaluation": evaluation,
        "network": {"allowed": f"{allowed[0]}:{allowed[1]}" if allowed else None,
                    "connections": guard.connections, "blocked": guard.blocked},
        "diagnostics": {"log_counts": dict(capture.counts), "log_records": capture.records,
                        "warnings": [f"{w.category.__name__}: {str(w.message)[:300]}" for w in caught[:MAX_DIAGNOSTICS]]},
    }
    _write_new(directory, report_name, _json_bytes(report))
    _print_summary(report, directory / report_name, ledger)
    return 1 if failure else 0


def _print_summary(report: dict[str, Any], path: Path, ledger: Path | None) -> None:
    """Сводка без текста документа: статусы, счётчики и ID строк."""
    print(f"Итог {report['task_name']}: {report['outcome']}"
          + (f" ({report['failure']['kind']})" if report["failure"] else "") + f"; проверка: {report['gate']}")
    for call in report["calls"]:
        print(f"  вызов #{call['index']} {call['outcome']} HTTP={call['http_status']} finish={call['finish_reason']} "
              f"{call['duration_s']} с usage={call['usage']}")
    result = report["result"]
    print(f"Промпт: {result['prompt_checks']}")
    print(f"Привязка: {result.get('grounding_counts')}; атрибуция: {result.get('attribution_counts')}; "
          f"дубли: {result.get('duplicates')}; перекрытий: {len(result.get('overlaps', []))}; "
          f"сырые→итог: {(result.get('raw_vs_final') or {}).get('raw_item_count')}→"
          f"{(result.get('raw_vs_final') or {}).get('final_item_count')}")
    for item in result.get("extractions", []):
        print(f"  [{item['position']}] {item['class']} {item['grounding']} {item['char_interval']} "
              f"строки={item['lines']} вхождений={item['exact_occurrences']}")
    for entry in result.get("resolved", []):
        print(f"  связь [{entry['position']}] {entry['source_lines']} → {entry['target_lines']} "
              f"{entry['relation_type']} {entry['resolution']} проблемы={entry['problems']}")
    evaluation = report["evaluation"]
    if "required_lines" in evaluation:
        print(f"Чек-лист: строки {evaluation['required_lines_covered']}/{evaluation['required_lines']}, "
              f"пропущены {evaluation['missing_required_lines']}, классы не совпали {evaluation['class_mismatches']}")
    if "required_total" in evaluation:
        print(f"Чек-лист связей: {evaluation['required_found']}/{evaluation['required_total']}, "
              f"FN {evaluation['false_negatives_required']}, "
              f"только с неверной гранулярностью {evaluation['found_only_with_wrong_granularity']}, "
              f"разрешены как ожидалось {[r['id'] for r in evaluation['expectations'] if r['resolved_as_expected']]}, "
              f"ошибки типа {evaluation['type_errors']}, "
              f"вне чек-листа {evaluation['unmatched_candidates']}")
    print(f"Сеть: соединений {report['network']['connections']}, заблокировано {len(report['network']['blocked'])}"
          + (f"; журнал POST: {_ledger_count(ledger)}" if ledger else ""))
    print(f"Статус: {report['legal_status']} / {report['review_status']}. Отчёт (частный): {path}")


# --- LE-3A: офлайн Evidence → ContextPackage → Claim -----------------------------------------------

OFFLINE_COMMANDS = ("evidence-prepare", "claim-request", "claim-check", "claim-run")


def _snapshot(args: argparse.Namespace) -> evidence.Snapshot:
    excerpt, source_map, _, _ = _load_inputs(args)
    try:
        return evidence.snapshot(excerpt, source_map)
    except evidence.EvidenceError as error:
        raise Refused(str(error)) from None


def _checked_package(args: argparse.Namespace, snap: evidence.Snapshot) -> tuple[dict[str, Any], list[Any]]:
    package = json.loads(_private_file(args.context))
    try:
        return package, evidence.check_package(snap, package)
    except evidence.EvidenceError as error:
        raise Refused(str(error)) from None


def evidence_prepare(args: argparse.Namespace) -> int:
    snap = _snapshot(args)
    selection = _private_file(args.selection, MAX_INPUT_BYTES)
    # Разбор отсылок LE-2 (references → task) транзитивно импортирует langextract; клиент модели не создаётся.
    from . import references

    try:
        package = evidence.build_package(snap, json.loads(selection), source.sha256(selection),
                                         references.parse_references(snap.excerpt))
        records = [item["evidence"] for item in package["evidence"]]
        full_txt = (evidence.verify_full_txt(snap, _private_file(args.full_txt, 50_000_000).decode("utf-8"), records)
                    if args.full_txt else None)
    except evidence.EvidenceError as error:
        raise Refused(str(error)) from None
    name = f"{package['context_package_id']}.context.json"
    directory = _directory(args.output_dir, (name,))
    _write_new(directory, name, _json_bytes(package))
    links = package["derived_links"]
    print(f"Снимок {snap.snapshot_id}; пакет {package['context_package_id']}; "
          f"полнота {package['context_completeness']}; смещения полного TXT сверены: {full_txt}")
    for item in package["evidence"]:
        record = item["evidence"]
        print(f"  {record['evidence_id']} {record['line_id']} {item['role']} "
              f"{'line' if record['whole_line'] else 'subspan'} {record['chars']}")
    print(f"Неразрешённые зависимости: {[(d['dependency_id'], d['kind'], d['line']) for d in package['unresolved_dependencies']]}")
    print(f"Связи с целью: {[(link['from_line'], link['granularity'], link['included']) for link in links['references_to_target_line']]}; "
          f"не включены, но ссылаются на цель: {links['omitted_lines_referencing_target']}")
    print(f"Отсылки из цели: {[(link['point'], link['in_excerpt'], link['referenced_lines_not_included']) for link in links['references_from_target_line']]}; "
          f"вне фрагмента: {[(entry['line'], entry['point']) for entry in links['included_lines_with_out_of_excerpt_references']]}; "
          f"не включены и не связаны: {links['not_included_not_linked']}")
    print(f"Статус: {package['legal_status']} / {package['review_status']}. Пакет (частный): {directory / name}")
    return 0


def claim_request(args: argparse.Namespace) -> int:
    snap = _snapshot(args)
    package, pairs = _checked_package(args, snap)
    request = claims.render_request(snap, package, pairs)
    name = f"{package['context_package_id']}.claim-request.json"
    directory = _directory(args.output_dir, (name,))
    _write_new(directory, name, _json_bytes(request))
    messages = request["body"]["messages"]
    print(f"Запрос для {package['context_package_id']}: сообщений {len(messages)}, символов {[len(m['content']) for m in messages]}, "
          f"доказательств {len(pairs)}, body SHA-256 {request['body_sha256']}; transport={request['transport']}")
    print(f"Ничего не отправлено. Запрос (частный): {directory / name}")
    return 0


def claim_check(args: argparse.Namespace) -> int:
    snap = _snapshot(args)
    package, pairs = _checked_package(args, snap)
    response = _private_file(args.response, MAX_INPUT_BYTES * 4)
    try:
        entries, rejection = claims.validate_response(response.decode("utf-8"), snap, package, pairs), None
    except claims.ClaimError as error:
        entries, rejection = [], str(error)
    except evidence.EvidenceError as error:
        raise Refused(str(error)) from None
    results = [entry["summary"] for entry in entries]
    name = f"{package['context_package_id']}.claim-check.{_utc()}.{time.monotonic_ns() % 1_000_000:06d}.json"
    full_name = name.removesuffix(".json") + ".claim-result.json" if args.full_result else None
    directory = _directory(args.output_dir, (name,) + ((full_name,) if full_name else ()))
    if full_name and not rejection:
        # Ответ из локального файла не доказан как ответ на конкретный запрос: привязка запроса не установлена.
        _write_new(directory, full_name, _json_bytes(claims.result_artifact(snap, package, entries, {
            "origin": "LOCAL_RESPONSE_FILE", "request_binding": "NOT_ESTABLISHED", "request_body_sha256": None,
            "payload_sha256": None, "model": None, "response_sha256": source.sha256(response),
            "content_sha256": source.sha256(response)})))
    report = {"contract": claims.CLAIM_CONTRACT, "context_package_id": package["context_package_id"],
              "response_sha256": source.sha256(response), "status": "REJECTED" if rejection else "ACCEPTED_AS_CANDIDATES",
              "rejection": rejection, "claims": results, "legal_fidelity": "NOT_VERIFIED",
              "legal_status": "UNVERIFIED", "review_status": "REVIEW_REQUIRED"}
    _write_new(directory, name, _json_bytes(report))
    print(f"Проверка ответа: {report['status']}" + (f" ({rejection})" if rejection else ""))
    for result in results:
        print(f"  [{result['position']}] {result['candidate_id']} строки={result['evidence_lines']} {result['verdicts']} "
              f"неразрешённых={len(result['unresolved_dependencies'])} "
              f"не отмечены моделью={len(result['package_dependencies_not_acknowledged'])}")
    if full_name:
        print(f"Полный результат (частный, с текстом утверждений): {full_name if not rejection else 'не записан (ответ отклонён)'}")
    print(f"Статус: UNVERIFIED / REVIEW_REQUIRED. Отчёт (частный): {directory / name}")
    return 1 if rejection else 0


# --- LE-3B.1: один вызов Claim Construction --------------------------------------------------------

def _claim_run_inputs(args: argparse.Namespace) -> tuple[evidence.Snapshot, dict[str, Any], list[Any],
                                                         dict[str, Any], dict[str, Any], str]:
    """Пакет, запрос и тело POST, заново построенные и сверенные с тем, что владелец закрепил при проверке."""
    if args.max_tokens != claims.MAX_TOKENS:
        raise Refused(f"--max-tokens должен равняться {claims.MAX_TOKENS}; бюджет меняется только согласованной правкой кода")
    snap = _snapshot(args)
    package, pairs = _checked_package(args, snap)
    if package["context_package_id"] != args.expect_context_package_id:
        raise Refused("ID пакета не совпадает с --expect-context-package-id")
    request = claims.render_request(snap, package, pairs)
    try:
        recorded = parse_json(_private_file(args.request).decode("utf-8"))
        claim_run.check_request(recorded, request)
        value = claim_run.payload(request, args.model)
    except (ValueError, claim_run.RunError) as error:
        raise Refused(str(error) if isinstance(error, claim_run.RunError) else "Файл запроса не является строгим JSON") from None
    if request["body_sha256"] != args.expect_body_sha256:
        raise Refused("body SHA-256 заново построенного запроса не совпадает с --expect-body-sha256")
    return snap, package, pairs, request, value, claim_run.sha256(claim_run.payload_bytes(value))


def run_claim(args: argparse.Namespace) -> int:
    """claim-run: replay / preflight без сети; live — не более одного POST после всех проверок."""
    snap, package, pairs, request, value, payload_sha256 = _claim_run_inputs(args)
    context_id, body_sha256 = package["context_package_id"], request["body_sha256"]
    mode = "replay" if args.replay else "preflight" if args.preflight else "live"
    config, allowed, ledger, exchange = None, None, None, None
    if mode == "replay":
        if args.endpoint or args.post_ledger or args.timeout is not None:
            raise Refused("В режиме --replay не задаются --endpoint, --post-ledger и --timeout")
        try:
            exchange = claim_run.load_exchange(_private_file(args.replay, claim_run.MAX_EXCHANGE_BYTES))
        except claim_run.RunError as error:
            raise Refused(str(error)) from None
        if (exchange["context_package_id"] != context_id or exchange["request_body_sha256"] != body_sha256
                or exchange["payload_sha256"] != payload_sha256 or exchange["model"] != args.model):
            raise Refused("Файл обмена записан для другого пакета, запроса или модели")
    else:
        if not (args.endpoint and args.post_ledger and args.post_cap is not None and args.timeout is not None
                and args.expect_ledger_count is not None):
            raise Refused("Для --preflight и --live нужны --endpoint, --post-ledger, --post-cap, "
                          "--expect-ledger-count и --timeout")
        config = LLMConfig(args.endpoint, args.model, api_key=os.environ.get("LLM_WIKI_API_KEY"),
                           timeout=args.timeout, response_format="json_object")
        try:
            allowed = _endpoint_target(config)
            ledger = claim_run.ledger_path(args.post_ledger)
            count = claim_run.read_ledger(ledger)
        except (ValueError, claim_run.RunError) as error:
            raise Refused(str(error)) from None
        if args.post_cap < 1 or count != args.expect_ledger_count or count + 1 > args.post_cap:
            raise Refused(f"Журнал POST: {count}, ожидалось {args.expect_ledger_count}, предел {args.post_cap}; "
                          "отправка невозможна")

    base = f"{context_id}.claim-run.{mode}.{_utc()}.{time.monotonic_ns() % 1_000_000:06d}"
    names = {"report": base + ".report.json", "result": base + ".claim-result.json",
             "exchange": base + ".exchange.json"}
    directory = _directory(args.output_dir, tuple(names.values()))
    if mode == "preflight":
        messages = value["messages"]
        print(f"Preflight {context_id}: снимок {snap.snapshot_id}; доказательств {len(pairs)}, "
              f"зависимостей {len(package['unresolved_dependencies'])}, полнота {package['context_completeness']}")
        print(f"body SHA-256 {body_sha256}; payload SHA-256 {payload_sha256}; сообщений {len(messages)}, "
              f"символов {[len(m['content']) for m in messages]}; max_tokens {value['max_tokens']}; "
              f"тайм-аут {config.timeout} с (на операцию сокета, не общий срок); предел ответа {claim_run.MAX_ENVELOPE_BYTES} байт")
        print(f"Журнал POST: {count} из {args.post_cap}; после live-вызова будет {count + 1}. "
              f"Ключ API задан: {config.api_key is not None}. Адрес и модель проверены по форме, сервер не опрашивался.")
        print("Ничего не отправлено и не записано. Разрешение на POST этим не выдаётся.")
        return 0

    removed = _clean_environment()
    guard = NetworkGuard(allowed if mode == "live" else None)
    sys.addaudithook(guard)
    if mode == "live":
        from experiments.langextract.model import http_transport

        send, url, timeout = http_transport(config), config.endpoint.rstrip("/") + "/chat/completions", config.timeout

        def reserve() -> tuple[int, int]:  # строка post — до отправки, под исключительной блокировкой журнала
            return claim_run.reserve_post(ledger, args.post_cap, args.expect_ledger_count, {
                "utc": datetime.now(timezone.utc).isoformat(), "task": "LE-3B", "context_package_id": context_id,
                "request_body_sha256": body_sha256, "payload_sha256": payload_sha256})
    else:
        send, url, timeout, reserve = claim_run.replay_transport(exchange, context_id, body_sha256), "replay:", 0.0, None
    try:
        attempt = claim_run.send_once(send, url, value, timeout, reserve)
    except claim_run.RunError as error:
        raise Refused(str(error)) from None

    raw, content, diagnostics, rejection, entries = attempt.pop("raw"), None, {}, None, []
    if attempt["transport_outcome"] != "ok":
        status = "TRANSPORT_FAILED"
    else:
        content, diagnostics, rejection = claim_run.envelope(raw)
        if attempt["http_status"] != 200:
            content, rejection = None, "Сервис вернул HTTP-статус, отличный от 200"
        status = "ENVELOPE_REJECTED" if rejection else "PENDING"
    if mode == "live":  # полный ответ сервиса — отдельный частный файл для аудита и replay, до разбора утверждений
        _write_new(directory, names["exchange"], _json_bytes(claim_run.exchange_record(
            context_id, body_sha256, payload_sha256, args.model, {**attempt, "raw": raw})))
    if content is not None:
        try:
            entries = claims.validate_response(content, snap, package, pairs)
            status = "ACCEPTED_AS_CANDIDATES"
        except (claims.ClaimError, evidence.EvidenceError) as error:
            status, rejection = "REJECTED", str(error)
    if status == "ACCEPTED_AS_CANDIDATES":
        _write_new(directory, names["result"], _json_bytes(claims.result_artifact(snap, package, entries, {
            "origin": f"CLAIM_RUN_{mode.upper()}", "request_binding": "REGENERATED_AND_MATCHED",
            "request_body_sha256": body_sha256, "payload_sha256": payload_sha256, "model": args.model,
            "response_sha256": diagnostics["response_sha256"], "content_sha256": diagnostics["content_sha256"]})))
    report = {
        "contract": claim_run.RUN_CONTRACT, "mode": mode, "context_package_id": context_id,
        "snapshot_id": snap.snapshot_id, "request_body_sha256": body_sha256, "payload_sha256": payload_sha256,
        "max_tokens": value["max_tokens"], "timeout_s": config.timeout if config else None,
        "envelope_limit_bytes": claim_run.MAX_ENVELOPE_BYTES, **attempt, "envelope": diagnostics,
        "status": status, "rejection": rejection, "claims": [entry["summary"] for entry in entries],
        "files": {"full_result": names["result"] if entries else None,
                  "exchange": names["exchange"] if mode == "live" else None},
        "network": {"policy": "ALLOW_VALIDATED_PRIVATE_DESTINATION" if mode == "live" else "BLOCK_ALL",
                    "connections": guard.connections, "blocked": len(guard.blocked)},
        "removed_environment_count": len(removed), "technical_acceptance_only": True,
        "legal_fidelity": "NOT_VERIFIED", "legal_status": "UNVERIFIED", "review_status": "REVIEW_REQUIRED",
    }
    _write_new(directory, names["report"], _json_bytes(report))
    print(f"claim-run {mode}: {status}" + (f" ({rejection})" if rejection else "")
          + f"; транспорт {attempt['transport_outcome']} HTTP={attempt['http_status']} "
          f"finish={diagnostics.get('finish_reason')} usage={diagnostics.get('usage')} {attempt['duration_s']} с")
    print(f"Журнал POST: {attempt['ledger_before_after']}; бюджет израсходован: {attempt['budget_consumed']}; повторов нет; "
          f"сеть: соединений {guard.connections}, заблокировано {len(guard.blocked)}")
    for result in report["claims"]:
        print(f"  [{result['position']}] {result['candidate_id']} строки={result['evidence_lines']} {result['verdicts']}")
    print(f"Статус: UNVERIFIED / REVIEW_REQUIRED. Файлы (частные): {directory.name}/"
          f"{{{', '.join(name for name in (names['report'], report['files']['full_result'], report['files']['exchange']) if name)}}}")
    return 0 if status == "ACCEPTED_AS_CANDIDATES" else 1


def _offline(args: argparse.Namespace, block_network: bool = True) -> int:
    """Команды LE-3A/LE-3B.1: сеть процесса блокируется (кроме claim-run --live); подробности ошибок не выводятся."""
    if block_network:
        sys.addaudithook(NetworkGuard(None))
    try:
        return args.handler(args)
    except Refused as error:
        print(f"Отказ: {error}", file=sys.stderr)
    except Exception as error:  # сообщение может содержать текст частного файла
        print(f"Отказ: внутренняя ошибка {type(error).__name__}; подробности скрыты", file=sys.stderr)
    return 1


def _claim_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request", required=True, help="claim-request того же пакета: должен совпасть с заново построенным")
    parser.add_argument("--expect-context-package-id", required=True, help="ID пакета, проверенного владельцем")
    parser.add_argument("--expect-body-sha256", required=True, help="body SHA-256 запроса, проверенного владельцем")
    parser.add_argument("--max-tokens", type=int, required=True,
                        help=f"подтверждение бюджета ответа; допустимо только {claims.MAX_TOKENS} (не повышается)")
    parser.add_argument("--model", required=True, help="точный ID одной модели")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", help="частный файл обмена le3b-claim-exchange/v1: воспроизвести без сети")
    mode.add_argument("--preflight", action="store_true", help="проверить условия live-вызова без сети и без записи")
    mode.add_argument("--live", action="store_true",
                      help="один реальный POST; только по отдельному свежему разрешению владельца")
    parser.add_argument("--endpoint", help="--preflight/--live: базовый URL с частным/loopback IP, включая /v1")
    parser.add_argument("--post-ledger", help="--preflight/--live: частный журнал POST (JSONL, 0600, в runtime)")
    parser.add_argument("--post-cap", type=int, help="--preflight/--live: предел накопленных POST по журналу")
    parser.add_argument("--expect-ledger-count", type=int,
                        help="--preflight/--live: число POST в журнале, которое владелец видел перед запуском")
    parser.add_argument("--timeout", type=float, help="--preflight/--live: тайм-аут операции сокета, 1–300 с")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.langextract_real", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare", help="офлайн: RTF → TXT → фрагмент и карта источника")
    make.add_argument("rtf", help="локальный закреплённый RTF")
    make.add_argument("--spec", required=True, help="частная спецификация источника (JSON в runtime)")
    make.add_argument("--output-dir", required=True, help="существующий игнорируемый каталог в runtime, 0700")
    make.add_argument("--libreoffice", help="путь к soffice; по умолчанию поиск в PATH")
    make.set_defaults(handler=prepare)
    check = commands.add_parser("probe", help="GET /health, /v1/models, /slots, /metrics без содержимого слотов")
    check.add_argument("--endpoint", required=True, help="одобренный базовый URL с частным IP, включая /v1")
    check.add_argument("--label", required=True, choices=("before", "during", "after"))
    check.add_argument("--output-dir", required=True, help="существующий игнорируемый каталог в runtime, 0700")
    check.set_defaults(handler=probe)
    go = commands.add_parser("run", help="одна задача LE-2A или LE-2B, не более одного вызова модели")
    go.add_argument("input", help="UTF-8 фрагмент, созданный prepare")
    go.add_argument("--source-map", required=True)
    go.add_argument("--task", required=True, choices=tuple(TASKS))
    go.add_argument("--output-dir", required=True, help="существующий игнорируемый каталог в runtime, 0700")
    go.add_argument("--checklist", help="частный чек-лист ожидаемых наблюдений (JSON), составленный до вызовов")
    go.add_argument("--le1-report", help="отчёт LE-1: сравнить часть промпта до текста с промптом LE-1")
    go.add_argument("--passages-report", help="для relations: отчёт LE-2A с проверкой PASS")
    go.add_argument("--endpoint")
    go.add_argument("--model", help="точный ID модели из /v1/models")
    go.add_argument("--post-ledger", help="частный журнал POST (JSONL с базовой строкой)")
    go.add_argument("--post-cap", type=int, default=10, help="предел накопленных POST по журналу")
    go.add_argument("--timeout", type=float, default=300.0, help="тайм-аут одного вызова, 1–300 с")
    go.add_argument("--replay", help="отчёт LE-2 той же задачи: воспроизвести ответы без сети")
    go.set_defaults(handler=run)
    for command, handler, help_text in (
            ("evidence-prepare", evidence_prepare, "LE-3A офлайн: манифест выбора → доказательства и ContextPackage"),
            ("claim-request", claim_request, "LE-3A офлайн: промпт Claim Construction в частный файл, без отправки"),
            ("claim-check", claim_check, "LE-3A офлайн: проверить ответ модельной формы; смысл не проверяется"),
            ("claim-run", run_claim, "LE-3B.1: один вызов Claim Construction; без --live — только replay/preflight без сети")):
        offline = commands.add_parser(command, help=help_text)
        offline.add_argument("input", help="UTF-8 фрагмент, созданный prepare")
        offline.add_argument("--source-map", required=True)
        offline.add_argument("--output-dir", required=True, help="существующий игнорируемый каталог в runtime, 0700")
        if command == "evidence-prepare":
            offline.add_argument("--selection", required=True, help="частный манифест выбора строк и ролей (JSON)")
            offline.add_argument("--full-txt", help="полный TXT из prepare: сверить смещения с учётом BOM")
        else:
            offline.add_argument("--context", required=True, help="ContextPackage, созданный evidence-prepare")
        if command == "claim-check":
            offline.add_argument("--response", required=True, help="JSON-ответ модельной формы (локальный файл)")
            offline.add_argument("--full-result", action="store_true",
                                 help="также записать полный частный результат (с текстом утверждений) при приёме")
        if command == "claim-run":
            _claim_run_arguments(offline)
        offline.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    if args.command in OFFLINE_COMMANDS:
        return _offline(args, block_network=not (args.command == "claim-run" and args.live))
    if args.command == "run" and args.task == "relations" and not args.replay and not args.passages_report:
        parser.error("для --task relations без --replay нужен --passages-report")
    try:
        return args.handler(args)
    except Refused as error:
        print(f"Отказ: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
