"""CLI диагностического эксперимента LE-1: LangExtract 1.7.1 на синтетической фикстуре.

Один запуск: LE-1A (дословные нормативные фрагменты) → LE-1B (кандидаты связей) → независимая
проверка привязки → один частный JSON-отчёт и краткая сводка без исходного текста.
Ничего не публикуется; все результаты — UNVERIFIED / REVIEW_REQUIRED.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import ipaddress
import json
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Any
from urllib.parse import urlsplit
import warnings

from llm_wiki.llm import LLMConfig
from llm_wiki.structure import StructureError, _output_directory, _write_new

from . import fixture

SCHEMA = "llm-wiki-memory/langextract-experiment/v1"
MAX_INPUT_BYTES = 65_536
MAX_DIAGNOSTICS = 50
REMOVED_ENV_PREFIXES = ("OPENAI_", "LANGEXTRACT_", "GEMINI_", "GOOGLE_", "AZURE_", "HF_")
PROXY_VARIABLES = ("http_proxy", "https_proxy", "all_proxy", "no_proxy", "ftp_proxy")


class Refused(Exception):
    """Отказ до вызова модели: небезопасный ввод, конфигурация или превышение бюджета."""


class NetworkGuard:
    """Audit hook процесса: только одобренный IP:port; это дополнительная защита, не песочница."""

    def __init__(self, allowed: tuple[str, int] | None) -> None:
        self.allowed = allowed
        self.connections = 0
        self.blocked: list[dict[str, str]] = []

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        if event == "socket.connect":
            address = args[1]
            if isinstance(address, tuple) and self.allowed and (str(address[0]), address[1]) == self.allowed:
                self.connections += 1
                return
            self._block(event, address[0] if isinstance(address, tuple) else "non-inet")
        elif event == "socket.getaddrinfo":
            host = args[0].decode() if isinstance(args[0], bytes) else str(args[0])
            if self.allowed and host == self.allowed[0] and str(args[1]) == str(self.allowed[1]):
                return
            self._block(event, host)

    def _block(self, event: str, target: Any) -> None:
        self.blocked.append({"event": event, "target": str(target)[:100]})
        raise PermissionError("Сетевое обращение заблокировано политикой эксперимента")


class _Capture(logging.Handler):
    """Журнал библиотек — только в частный отчёт (сообщения могут содержать фрагменты текста)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[dict[str, str]] = []
        self.counts: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        self.counts[f"{record.name}:{record.levelname}"] += 1
        if len(self.records) < MAX_DIAGNOSTICS:
            self.records.append({"logger": record.name, "level": record.levelname,
                                 "message": record.getMessage()[:500]})


def _clean_environment() -> list[str]:
    removed = sorted(name for name in os.environ
                     if name.startswith(REMOVED_ENV_PREFIXES) or name.lower() in PROXY_VARIABLES)
    for name in removed:
        del os.environ[name]
    os.environ["LANGEXTRACT_DISABLE_PLUGINS"] = "1"
    return removed


def _read_input(raw_path: str, approved_source: bool) -> tuple[str, dict[str, Any]]:
    if "://" in raw_path:
        raise Refused("Вход должен быть локальным файлом; URL не принимаются")
    path = Path(raw_path)
    if not path.is_file():
        raise Refused("Входной файл не найден или не является обычным файлом")
    data = path.read_bytes()
    if len(data) > MAX_INPUT_BYTES:
        raise Refused(f"Входной файл больше {MAX_INPUT_BYTES} байт; обрезка не выполняется")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise Refused("Входной файл не является UTF-8") from None
    digest = hashlib.sha256(data).hexdigest()
    is_fixture = digest == fixture.FIXTURE_SHA256
    if not is_fixture and not approved_source:
        raise Refused("Вход не совпадает с синтетической фикстурой; для иного текста нужен явный --approved-source")
    return text, {"file_name": path.name, "size_bytes": len(data), "sha256": digest, "chars": len(text),
                  "is_le1_fixture": is_fixture, "offsets": "символы Unicode точного UTF-8 текста, [start, end)",
                  "original_document_fidelity": "NOT_APPLICABLE_SYNTHETIC" if is_fixture else "NOT_VERIFIED"}


def _endpoint_target(config: LLMConfig) -> tuple[str, int]:
    config.validate()
    url = urlsplit(config.endpoint)
    host = str(ipaddress.ip_address(url.hostname or ""))
    return host, url.port or (443 if url.scheme == "https" else 80)


def _questions(prompt: str) -> str | None:
    """Текст чанка в промпте LangExtract: после последнего «\\nQ: » до завершающего «\\nA: »."""
    marker, suffix = "\nQ: ", "\nA: "
    if marker not in prompt or not prompt.endswith(suffix):
        return None
    return prompt[prompt.rindex(marker) + len(marker):-len(suffix)]


def _run_task(lx: Any, spec: Any, text: str, model: Any, args: argparse.Namespace,
              passes: int) -> tuple[Any, dict[str, Any] | None]:
    from langextract import prompt_validation as pv
    from langextract import resolver

    from .model import CallFailed

    resolver_params: dict[str, Any] = {"suppress_parse_errors": False}
    if args.alignment == "strict":
        resolver_params.update(enable_fuzzy_alignment=False, accept_match_lesser=False)
    model.task = spec.name
    try:
        document = lx.extract(
            text, prompt_description=spec.prompt, examples=list(spec.examples), model=model,
            fence_output=False, use_schema_constraints=False, max_char_buffer=args.max_char_buffer,
            extraction_passes=passes, batch_length=1, max_workers=1, context_window_chars=None,
            resolver_params=resolver_params, show_progress=False, debug=False,
            prompt_validation_level=pv.PromptValidationLevel.ERROR, prompt_validation_strict=True,
        )
    except CallFailed:
        return None, {"task": spec.name, "kind": model.calls[-1].outcome if model.calls else "call_failed"}
    except resolver.ResolverParsingError as error:
        return None, {"task": spec.name, "kind": "invalid_model_output", "detail": str(error)[:300]}
    except pv.PromptAlignmentError:
        return None, {"task": spec.name, "kind": "example_alignment_error"}
    except Exception as error:  # Сбой библиотеки фиксируется как вид без трассировки и текста.
        return None, {"task": spec.name, "kind": "library_error", "type": type(error).__name__}
    return document, None


def run(args: argparse.Namespace) -> int:
    text, source = _read_input(args.input, args.approved_source)
    replay = None
    if args.replay:
        replay = json.loads(Path(args.replay).read_text(encoding="utf-8"))
        if replay.get("schema") != SCHEMA:
            raise Refused("Файл --replay не является отчётом этого эксперимента")
        if replay["source"]["sha256"] != source["sha256"]:
            raise Refused("Отчёт --replay записан для другого входа")
        recorded = replay["configuration"]
        config = LLMConfig(recorded["endpoint"], recorded["model"], timeout=recorded["timeout_s"],
                           response_format=recorded["response_format"])
        allowed = None
    else:
        if not args.endpoint or not args.model:
            raise Refused("Для реального запуска нужны --endpoint и --model (или --replay)")
        config = LLMConfig(args.endpoint, args.model, api_key=os.environ.get("LLM_WIKI_API_KEY"),
                           timeout=args.timeout, response_format=args.response_format)
        try:
            allowed = _endpoint_target(config)
        except ValueError as error:
            raise Refused(str(error)) from None

    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_name = (f"{Path(source['file_name']).stem}.{'replay' if replay else 'http'}."
                   f"b{args.max_char_buffer}-p{args.passes}-{args.alignment}.{created}.le1-report.json")
    try:
        directory = _output_directory(Path(args.output_dir), (report_name,))
    except StructureError as error:
        raise Refused(str(error)) from None

    removed = _clean_environment()
    guard = NetworkGuard(allowed)
    sys.addaudithook(guard)
    capture = _Capture()
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.WARNING)

    import langextract as lx
    from langextract import chunking
    from langextract.core import tokenizer

    from . import grounding
    from .model import GuardedChatModel, call_summary, http_transport, replay_transport, to_dict
    from .task import PASSAGES, RELATIONS

    regex_tokenizer = tokenizer.RegexTokenizer()
    planned_chunks = list(chunking.ChunkIterator(
        text=regex_tokenizer.tokenize(text), max_char_buffer=args.max_char_buffer, tokenizer_impl=regex_tokenizer))
    chunks = [chunk.chunk_text for chunk in planned_chunks]
    intervals = [[chunk.char_interval.start_pos, chunk.char_interval.end_pos] for chunk in planned_chunks]
    planned = {PASSAGES.name: len(chunks) * args.passes, RELATIONS.name: len(chunks)}
    if sum(planned.values()) > args.max_calls:
        raise Refused(f"План требует {sum(planned.values())} вызовов, лимит --max-calls {args.max_calls}")

    transport = replay_transport(replay["calls"]) if replay else http_transport(config)
    model = GuardedChatModel(config, transport, max_calls=args.max_calls,
                             max_tokens=args.max_tokens, temperature=args.temperature)
    results: dict[str, Any] = {}
    failure = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for spec, passes in ((PASSAGES, args.passes), (RELATIONS, 1)):
            document, failure = _run_task(lx, spec, text, model, args, passes)
            if failure:
                break
            results[spec.name] = document

    tasks: dict[str, Any] = {}
    grounded: dict[str, list[dict[str, Any]]] = {}
    for spec, kind in ((PASSAGES, "passage"), (RELATIONS, "relation")):
        task_calls = [call for call in model.calls if call.task == spec.name]
        expected = chunks * (args.passes if spec is PASSAGES else 1)
        questions = [_questions(call.prompt) for call in task_calls]
        raw = [{"call": call.index, "pass": number // len(chunks) + 1, "chunk": number % len(chunks) + 1,
                **(grounding.raw_items(call.content, spec.classes) if call.content is not None
                   else {"strict_parse": "NO_CONTENT", "items": []})}
               for number, call in enumerate(task_calls)]
        entry: dict[str, Any] = {
            "status": "COMPLETED" if spec.name in results else ("FAILED" if failure and failure["task"] == spec.name
                                                                 else "NOT_RUN"),
            "planned_calls": planned[spec.name], "observed_calls": len(task_calls),
            "prompt_chunk_matches_plan": [q == e for q, e in zip(questions, expected)],
            "fenced_prompt": any("```" in call.prompt for call in task_calls),
            "raw_per_call": raw,
        }
        if spec.name in results:
            items = grounding.ground(text, source["sha256"], results[spec.name].extractions or [], kind)
            grounded[spec.name] = items
            entry.update(
                extractions=items,
                grounding_counts=dict(Counter(item["grounding"] for item in items)),
                library_alignment_counts=dict(Counter(str(item["library_alignment_status"]) for item in items)),
                duplicates=[item["position"] for item in items if item["duplicate_of"] is not None],
                overlaps=grounding.overlaps(items),
                same_span_different_class=grounding.same_span_different_class(items),
                raw_vs_final=grounding.raw_vs_final(raw, items),
            )
        tasks[spec.name] = entry

    evaluation: dict[str, Any] = {}
    if PASSAGES.name in grounded and RELATIONS.name in grounded:
        tasks[RELATIONS.name]["resolved"] = grounding.resolve_relations(
            text, grounded[RELATIONS.name], grounded[PASSAGES.name])
    if source["is_le1_fixture"] and PASSAGES.name in grounded:
        evaluation["passages"] = grounding.fixture_coverage(text, grounded[PASSAGES.name])
        if "resolved" in tasks[RELATIONS.name]:
            evaluation["link_5_to_1"] = grounding.expected_link(tasks[RELATIONS.name]["resolved"])

    report = {
        "schema": SCHEMA, "created_utc": created,
        "outcome": "PIPELINE_FAILED" if failure else "PIPELINE_COMPLETED", "failure": failure,
        "legal_status": "UNVERIFIED", "review_status": "REVIEW_REQUIRED",
        "versions": {"python": platform.python_version(),
                     **{name: importlib.metadata.version(name) for name in ("langextract", "openai")}},
        "source": source,
        "configuration": {
            "mode": "replay" if replay else "http", "replayed_from": Path(args.replay).name if replay else None,
            "endpoint": config.endpoint, "model": config.model, "provider": "GuardedChatModel (lx.extract(model=...))",
            "response_format": config.response_format, "timeout_s": config.timeout,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature if args.temperature is not None else "not sent (server default)", "max_calls": args.max_calls,
            "max_char_buffer": args.max_char_buffer, "passes_le1a": args.passes, "passes_le1b": 1,
            "context_window_chars": None, "alignment": args.alignment, "batch_length": 1, "max_workers": 1,
            "fence_output": False, "use_schema_constraints": False, "suppress_parse_errors": False,
            "prompt_validation": "ERROR, strict", "removed_environment": removed,
        },
        "chunks": {"count": len(chunks), "intervals": intervals, "full_context": len(chunks) == 1},
        "calls": [to_dict(call) for call in model.calls],
        "tasks": tasks, "evaluation": evaluation,
        "network": {"allowed": f"{allowed[0]}:{allowed[1]}" if allowed else None,
                    "connections": guard.connections, "blocked": guard.blocked},
        "diagnostics": {"log_counts": dict(capture.counts), "log_records": capture.records,
                        "warnings": [f"{w.category.__name__}: {str(w.message)[:300]}" for w in caught[:MAX_DIAGNOSTICS]]},
    }
    _write_new(directory, report_name, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    _print_summary(report, call_summary(model.calls), directory / report_name)
    return 1 if failure else 0


def _print_summary(report: dict[str, Any], calls: list[dict[str, Any]], path: Path) -> None:
    print(f"Итог: {report['outcome']}" + (f" ({report['failure']['task']}: {report['failure']['kind']})"
                                         if report["failure"] else ""))
    print(f"Чанки: {report['chunks']['count']} {report['chunks']['intervals']}; вызовов: {len(calls)}")
    for call in calls:
        print(f"  #{call['index']} {call['task']} {call['outcome']} finish={call['finish_reason']} "
              f"{call['duration_s']} с, usage={call['usage']}")
    for name, task in report["tasks"].items():
        print(f"{name}: {task['status']}; чанки промптов по плану: {task['prompt_chunk_matches_plan']}; "
              f"привязка: {task.get('grounding_counts')}; дубли: {task.get('duplicates')}; "
              f"перекрытий: {len(task.get('overlaps', []))}")
    passages = report["evaluation"].get("passages")
    if passages:
        print(f"Покрытие положений: {passages['provisions_with_verified_passage']}/{passages['provisions_total']}, "
              f"пропущены: {passages['missing_provisions']}, слияния: {len(passages['passages_spanning_several_provisions'])}, "
              f"6≠7: {passages['provisions_6_7_distinct']}, классы п.5: {passages['provision_5_classes']}")
    if "link_5_to_1" in report["evaluation"]:
        print(f"Связь 5 → 1: {report['evaluation']['link_5_to_1']}")
    print(f"Сеть: соединений {report['network']['connections']}, заблокировано {len(report['network']['blocked'])}")
    print(f"Статус: {report['legal_status']} / {report['review_status']}. Отчёт (частный): {path}")


def write_fixture(args: argparse.Namespace) -> int:
    name = f"{fixture.FIXTURE_ID}.txt"
    try:
        directory = _output_directory(Path(args.output_dir), (name,))
    except StructureError as error:
        raise Refused(str(error)) from None
    _write_new(directory, name, fixture.FIXTURE_BYTES)
    print(f"Фикстура: {directory / name}; {len(fixture.FIXTURE_BYTES)} байт; SHA-256 {fixture.FIXTURE_SHA256}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.langextract", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("fixture", help="записать синтетическую фикстуру LE-1")
    make.add_argument("--output-dir", required=True, help="существующий игнорируемый каталог в runtime, 0700")
    make.set_defaults(handler=write_fixture)
    go = commands.add_parser("run", help="один диагностический запуск LE-1A + LE-1B")
    go.add_argument("input", help="локальный UTF-8 TXT (по умолчанию принимается только фикстура LE-1)")
    go.add_argument("--output-dir", required=True, help="существующий игнорируемый каталог в runtime, 0700")
    go.add_argument("--endpoint", help="одобренный базовый URL с частным/loopback IP, включая /v1")
    go.add_argument("--model", help="точный ID модели из /v1/models")
    go.add_argument("--replay", help="отчёт прежнего запуска: воспроизвести записанные ответы без сети")
    go.add_argument("--approved-source", action="store_true",
                    help="явное разрешение обрабатывать текст, отличный от синтетической фикстуры")
    go.add_argument("--timeout", type=float, default=300.0, help="тайм-аут одного вызова, 1–300 с")
    go.add_argument("--max-calls", type=int, default=2, help="жёсткий лимит вызовов модели за запуск")
    go.add_argument("--max-char-buffer", type=int, default=4000,
                    help="max_char_buffer LangExtract; больше длины текста — один чанк")
    go.add_argument("--passes", type=int, default=1, choices=range(1, 4), help="extraction_passes для LE-1A")
    go.add_argument("--max-tokens", type=int, default=8192)
    go.add_argument("--temperature", type=float, default=None,
                    help="по умолчанию не отправляется (температура сервера), как в Docling Graph")
    go.add_argument("--response-format", choices=("json_object", "text"), default="json_object")
    go.add_argument("--alignment", choices=("library-default", "strict"), default="library-default",
                    help="strict отключает нечёткое выравнивание и MATCH_LESSER")
    go.set_defaults(handler=run)
    args = parser.parse_args(argv)
    if args.command == "run" and (args.max_calls < 1 or args.max_char_buffer < 50 or args.max_tokens < 1):
        parser.error("--max-calls ≥ 1, --max-char-buffer ≥ 50, --max-tokens ≥ 1")
    try:
        return args.handler(args)
    except Refused as error:
        print(f"Отказ: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
