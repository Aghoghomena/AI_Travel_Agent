"""
Evaluation Runner

Runs all evaluation suites and prints a grouped results table.

Usage:
    python -m src.evaluation.runner              # unit tests only
    python -m src.evaluation.runner --all        # unit + LLM integration tests
    python -m src.evaluation.runner --suite plan # single suite
"""

import sys
import time
import tempfile
import shutil
import os
import traceback
from pathlib import Path
from typing import Callable
from dataclasses import dataclass, field


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    name: str
    passed: bool
    duration_ms: float
    error: str = ""


@dataclass
class SuiteResult:
    suite_name: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self):  return sum(1 for c in self.cases if c.passed)
    @property
    def failed(self):  return sum(1 for c in self.cases if not c.passed)
    @property
    def total(self):   return len(self.cases)
    @property
    def pass_rate(self):
        return (self.passed / self.total * 100) if self.total else 0.0


# ── Fixture helpers ────────────────────────────────────────────────────────────

class _FixtureContext:
    """Minimal fixture runner that handles tmp_db and tmp_chroma."""

    def __init__(self):
        self._tmp_dirs: list[str] = []
        self._patches: list[tuple] = []

    def setup_tmp_db(self) -> str:
        d = tempfile.mkdtemp()
        self._tmp_dirs.append(d)
        db_file = os.path.join(d, "test.db")
        os.environ["SQLITE_DB_PATH"] = db_file

        import src.memory.db as db_mod
        db_mod.DB_PATH = db_file
        db_mod.initialize_db()
        return db_file

    def setup_tmp_chroma(self) -> str:
        d = tempfile.mkdtemp()
        self._tmp_dirs.append(d)
        chroma_dir = os.path.join(d, "chroma")
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)

        import src.memory.semantic as sem_mod
        sem_mod.CHROMA_PATH = chroma_dir
        return chroma_dir

    def teardown(self):
        for d in self._tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._tmp_dirs.clear()
        if "SQLITE_DB_PATH" in os.environ:
            del os.environ["SQLITE_DB_PATH"]


def _run_with_fixtures(fn: Callable, fx: _FixtureContext) -> None:
    """Calls fn, injecting fixture arguments it declares."""
    import inspect
    params = list(inspect.signature(fn).parameters.keys())

    kwargs = {}
    if "tmp_db" in params:
        kwargs["tmp_db"] = fx.setup_tmp_db()
    if "tmp_chroma" in params:
        kwargs["tmp_chroma"] = fx.setup_tmp_chroma()
    if "monkeypatch" in params:
        # Provide a lightweight monkeypatch shim
        class _MP:
            def __init__(self):  self._patches = []
            def setattr(self, obj, name, val):
                self._patches.append((obj, name, getattr(obj, name, None)))
                setattr(obj, name, val)
            def setenv(self, key, val):
                os.environ[key] = val
            def undo(self):
                for obj, name, original in reversed(self._patches):
                    if original is None:
                        try: delattr(obj, name)
                        except AttributeError: pass
                    else:
                        setattr(obj, name, original)
        mp = _MP()
        kwargs["monkeypatch"] = mp
        try:
            fn(**kwargs)
        finally:
            mp.undo()
        return

    fn(**kwargs)


def _run_case(name: str, fn: Callable) -> CaseResult:
    fx = _FixtureContext()
    start = time.perf_counter()
    try:
        _run_with_fixtures(fn, fx)
        duration = (time.perf_counter() - start) * 1000
        return CaseResult(name=name, passed=True, duration_ms=duration)
    except Exception as exc:
        duration = (time.perf_counter() - start) * 1000
        return CaseResult(
            name=name,
            passed=False,
            duration_ms=duration,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        fx.teardown()


# ── Suite runners ──────────────────────────────────────────────────────────────

def run_plan_suite() -> SuiteResult:
    from src.evaluation.eval_plan import CASES
    suite = SuiteResult("Plan Generation & Validation")
    for name, fn in CASES:
        suite.cases.append(_run_case(name, fn))
    return suite


def run_memory_suite() -> SuiteResult:
    from src.evaluation.eval_memory import CASES
    suite = SuiteResult("Long-Term Memory (Episodic / Procedural / Semantic)")
    for name, fn in CASES:
        suite.cases.append(_run_case(name, fn))
    return suite


def run_ranker_unit_suite() -> SuiteResult:
    from src.evaluation.eval_ranker import UNIT_CASES
    suite = SuiteResult("Ranker (Unit — no LLM)")
    for name, fn in UNIT_CASES:
        suite.cases.append(_run_case(name, fn))
    return suite


def run_ranker_integration_suite() -> SuiteResult:
    from src.evaluation.eval_ranker import INTEGRATION_CASES
    suite = SuiteResult("Ranker (Integration — LLM)")
    for name, fn in INTEGRATION_CASES:
        suite.cases.append(_run_case(name, fn))
    return suite


def run_intent_suite() -> SuiteResult:
    from src.evaluation.eval_intent import CASES
    suite = SuiteResult("Intent Classification & Elicitation (LLM)")
    for name, fn in CASES:
        suite.cases.append(_run_case(name, fn))
    return suite


# ── Display ────────────────────────────────────────────────────────────────────

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _tick(passed: bool) -> str:
    return f"{_GREEN}✓{_RESET}" if passed else f"{_RED}✗{_RESET}"


def print_suite(suite: SuiteResult) -> None:
    print(f"\n{_BOLD}{'─' * 70}{_RESET}")
    print(f"{_BOLD}{suite.suite_name}{_RESET}")
    print(f"{'─' * 70}")
    for c in suite.cases:
        status = _tick(c.passed)
        duration = f"{c.duration_ms:6.1f}ms"
        print(f"  {status}  {c.name:<52}  {duration}")
        if not c.passed and c.error:
            short_err = c.error.split("\n")[0][:80]
            print(f"       {_RED}{short_err}{_RESET}")
    bar_color = _GREEN if suite.failed == 0 else _RED
    print(f"{'─' * 70}")
    print(f"  {bar_color}{suite.passed}/{suite.total} passed{_RESET}   pass rate: {bar_color}{suite.pass_rate:.0f}%{_RESET}\n")


def print_summary(suites: list[SuiteResult]) -> None:
    total_passed = sum(s.passed for s in suites)
    total_cases  = sum(s.total  for s in suites)
    overall_rate = (total_passed / total_cases * 100) if total_cases else 0

    print(f"\n{_BOLD}{'═' * 70}{_RESET}")
    print(f"{_BOLD}EVALUATION SUMMARY{_RESET}")
    print(f"{'═' * 70}")
    print(f"  {'Suite':<45}  {'Pass':>5}  {'Fail':>5}  {'Rate':>6}")
    print(f"  {'─' * 45}  {'─' * 5}  {'─' * 5}  {'─' * 6}")
    for s in suites:
        color = _GREEN if s.failed == 0 else _RED
        print(
            f"  {s.suite_name:<45}  "
            f"{color}{s.passed:>5}{_RESET}  "
            f"{color}{s.failed:>5}{_RESET}  "
            f"{color}{s.pass_rate:>5.0f}%{_RESET}"
        )
    overall_color = _GREEN if total_passed == total_cases else _RED
    print(f"  {'─' * 45}  {'─' * 5}  {'─' * 5}  {'─' * 6}")
    print(
        f"  {'TOTAL':<45}  "
        f"{overall_color}{total_passed:>5}{_RESET}  "
        f"{overall_color}{total_cases - total_passed:>5}{_RESET}  "
        f"{overall_color}{overall_rate:>5.0f}%{_RESET}"
    )
    print(f"{'═' * 70}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def run_all(include_llm: bool = False, suite_filter: str | None = None) -> list[SuiteResult]:
    suites: list[SuiteResult] = []

    runners = {
        "plan":   run_plan_suite,
        "memory": run_memory_suite,
        "ranker": run_ranker_unit_suite,
    }
    llm_runners = {
        "ranker-llm": run_ranker_integration_suite,
        "intent":     run_intent_suite,
    }

    target = {**runners, **(llm_runners if include_llm else {})}

    if suite_filter:
        target = {k: v for k, v in target.items() if suite_filter in k}
        if not target:
            print(f"No suites match filter '{suite_filter}'. Available: {list({**runners, **llm_runners}.keys())}")
            return []

    for name, runner_fn in target.items():
        print(f"\n{_YELLOW}Running suite: {name}...{_RESET}", flush=True)
        suite = runner_fn()
        suites.append(suite)
        print_suite(suite)

    print_summary(suites)
    return suites


if __name__ == "__main__":
    include_llm   = "--all" in sys.argv
    suite_filter  = None
    for arg in sys.argv[1:]:
        if arg.startswith("--suite="):
            suite_filter = arg.split("=", 1)[1]
        elif arg == "--suite" and sys.argv.index(arg) + 1 < len(sys.argv):
            suite_filter = sys.argv[sys.argv.index(arg) + 1]

    results = run_all(include_llm=include_llm, suite_filter=suite_filter)
    failed = sum(s.failed for s in results)
    sys.exit(1 if failed else 0)
