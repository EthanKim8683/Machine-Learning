"""Observation-action tracer built on per-code ``sys.monitoring`` events."""

from __future__ import annotations

import dis
import io
import linecache
import os
import sys
import threading
import types
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict

# Contract annotates BaseSpan.type as types.FrameType.
_saved_config = BaseModel.model_config
BaseModel.model_config = ConfigDict(arbitrary_types_allowed=True)
try:
    from tracer_contract import (  # noqa: E402
        CallSpan,
        ExceptionSpan,
        LineSpan,
        ReturnSpan,
        Trace as ContractTrace,
        Tracer as TracerContract,
    )
finally:
    BaseModel.model_config = _saved_config

__all__ = [
    "CallSpan",
    "ExceptionSpan",
    "LineSpan",
    "ReturnSpan",
    "Trace",
    "Tracer",
]

TRACER_FILE = os.path.abspath(__file__)
USER_FILENAME = "<user>"
_TOOL_NAME = "cwm-tracer"
_TOOL_IDS = (3, 4, 2, 1, 5, 0)
_LOCAL_EVENTS = (
    sys.monitoring.events.LINE
    | sys.monitoring.events.PY_START
    | sys.monitoring.events.PY_RETURN
)

_SKIP_MARKERS = (
    "/contextlib.py",
    "/_pytest/",
    "/pytest/",
    "/pluggy/",
    "/site-packages/",
    "/dist-packages/",
    "/IPython/",
    "/ipykernel/",
    "/jupyter_client/",
    "/traitlets/",
    "/prompt_toolkit/",
)

_OMIT_TYPES = (
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    types.MethodType,
    types.CodeType,
    types.FrameType,
    types.TracebackType,
    types.GeneratorType,
    types.CoroutineType,
    type,
)

_IGNORE_NAMES = frozenset(
    {
        "__annotations__",
        "__builtins__",
        "__cached__",
        "__class__",
        "__dict__",
        "__doc__",
        "__exception__",
        "__file__",
        "__loader__",
        "__module__",
        "__name__",
        "__package__",
        "__return__",
        "__spec__",
        "__stdout__",
        "__weakref__",
    }
)

_STORE_OPS = {
    "STORE_DEREF",
    "STORE_FAST",
    "STORE_FAST_MAYBE_NULL",
    "STORE_GLOBAL",
    "STORE_NAME",
}

Span = CallSpan | LineSpan | ReturnSpan | ExceptionSpan


class Trace(ContractTrace):
    spans: list[Span]


class Tracer(TracerContract):
    """Record spans for the duration of ``with tracer.trace():``."""

    def __init__(self, max_spans: int | None = None) -> None:
        self._max_spans = max_spans
        self._traces: list[Trace] = []
        self._spans: list[Span] = []
        self._pending: dict[int, _PendingLine] = {}
        self._entry: set[tuple[int, int]] = set()
        self._armed: set[types.CodeType] = set()
        self._user_filename = ""
        self._tool_id: int | None = None
        self._thread = 0
        self._live = False
        self._busy = False

    @contextmanager
    def trace(self) -> Iterator[Tracer]:
        out, err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        self._spans = []
        self._pending = {}
        self._busy = False
        thread = threading.get_ident()
        try:
            sys.stdout = _Capture(old_out, out, thread)
            sys.stderr = _Capture(old_err, err, thread)
            self._start()
            yield self
        except BaseException as exc:
            self._record_exception(exc)
            raise
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            self._stop()
            self._traces.append(
                Trace(spans=self._spans, stdout=out.getvalue(), stderr=err.getvalue())
            )

    def trace_str(self, src: str) -> Trace:
        source = src if src.endswith("\n") else f"{src}\n"
        compiled = compile(source, USER_FILENAME, "exec")
        _prime_linecache(USER_FILENAME, source)
        previous, self._user_filename = self._user_filename, USER_FILENAME
        try:
            with self.trace():
                self._arm(compiled)
                exec(compiled, {"__name__": "__main__"})  # noqa: S102
        finally:
            self._user_filename = previous
        return self._traces[-1]

    def traces(self) -> list[Trace]:
        return list(self._traces)

    def clear(self) -> None:
        self._traces.clear()

    def _start(self) -> None:
        self._entry = set()
        self._armed = set()
        self._thread = threading.get_ident()
        tool_id = self._tool_id = _acquire_tool_id()
        events = sys.monitoring.events
        sys.monitoring.register_callback(tool_id, events.LINE, self._on_line)
        sys.monitoring.register_callback(tool_id, events.PY_START, self._on_start)
        sys.monitoring.register_callback(tool_id, events.PY_RETURN, self._on_return)
        sys.monitoring.set_events(tool_id, events.NO_EVENTS)
        frame = sys._getframe()
        while frame is not None:
            if self._wanted(frame.f_code):
                self._entry.add((id(frame), frame.f_lineno))
                self._arm(frame.f_code)
            frame = frame.f_back
        self._live = True

    def _stop(self) -> None:
        self._live = False
        try:
            self._flush_pending()
        except Exception:
            self._pending.clear()
        self._disarm()

    def _disarm(self) -> None:
        tool_id = self._tool_id
        if tool_id is None:
            return
        events = sys.monitoring.events
        sys.monitoring.set_events(tool_id, events.NO_EVENTS)
        none = events.NO_EVENTS
        for code in self._armed:
            try:
                sys.monitoring.set_local_events(tool_id, code, none)
            except ValueError:
                pass
        self._armed.clear()
        sys.monitoring.register_callback(tool_id, events.LINE, None)
        sys.monitoring.register_callback(tool_id, events.PY_START, None)
        sys.monitoring.register_callback(tool_id, events.PY_RETURN, None)
        try:
            sys.monitoring.free_tool_id(tool_id)
        except ValueError:
            pass
        self._tool_id = None

    def _arm(self, code: types.CodeType) -> None:
        tool_id = self._tool_id
        if tool_id is None or code in self._armed:
            return
        sys.monitoring.set_local_events(tool_id, code, _LOCAL_EVENTS)
        self._armed.add(code)
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                self._arm(const)

    def _wanted(self, code: types.CodeType) -> bool:
        filename = code.co_filename
        if self._user_filename:
            return filename == self._user_filename
        if filename.startswith("<frozen "):
            return False
        try:
            path = os.path.abspath(filename).replace("\\", "/")
        except (OSError, ValueError):
            path = filename.replace("\\", "/")
        if path == TRACER_FILE:
            return False
        if any(marker in path for marker in _SKIP_MARKERS):
            return False
        if not filename.startswith("<"):
            for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix):
                if prefix and path.startswith(
                    os.path.abspath(prefix).replace("\\", "/")
                ):
                    return False
        return True

    def _frame(self, code: types.CodeType) -> types.FrameType | None:
        if not self._live or self._busy or threading.get_ident() != self._thread:
            return None
        if not self._wanted(code):
            return None
        return _frame_for_code(code)

    def _on_line(self, code: types.CodeType, line_number: int) -> Any:
        frame = self._frame(code)
        if frame is None:
            return None
        try:
            self._see_line(frame, line_number)
        except (RecursionError, Exception):
            self._stop()
        return None

    def _on_start(self, code: types.CodeType, _offset: int) -> Any:
        frame = self._frame(code)
        if frame is None:
            return None
        try:
            self._see_call(frame)
        except (RecursionError, Exception):
            self._stop()
        return None

    def _on_return(self, code: types.CodeType, _offset: int, retval: Any) -> Any:
        frame = self._frame(code)
        if frame is None:
            return None
        try:
            self._see_return(frame, retval)
        except (RecursionError, Exception):
            self._stop()
        return None

    def _see_line(self, frame: types.FrameType, line_number: int) -> None:
        self._flush_frame(frame)
        if self._full() or (id(frame), line_number) in self._entry:
            return
        self._pending[id(frame)] = _PendingLine(
            frame=frame,
            line_number=line_number,
            line=_source(frame.f_code.co_filename, line_number),
        )

    def _see_call(self, frame: types.FrameType) -> None:
        if frame.f_code.co_name == "<module>":
            return
        if frame.f_back is not None:
            self._flush_frame(frame.f_back)
        self._add(
            CallSpan(
                type="call",
                line_number=frame.f_code.co_firstlineno,
                line=_source(frame.f_code.co_filename, frame.f_code.co_firstlineno),
                arguments=self._snapshot(frame.f_locals),
            )
        )

    def _see_return(self, frame: types.FrameType, retval: Any) -> None:
        self._flush_frame(frame)
        if frame.f_code.co_name == "<module>":
            return
        self._add(
            ReturnSpan(
                type="return",
                line_number=frame.f_lineno,
                line=_source(frame.f_code.co_filename, frame.f_lineno),
                return_value=self._format(retval),
            )
        )

    def _record_exception(self, exc: BaseException) -> None:
        if isinstance(exc, GeneratorExit):
            return
        traceback = exc.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if self._wanted(frame.f_code):
                self._flush_frame(frame)
                self._add(
                    ExceptionSpan(
                        type="exception",
                        line_number=frame.f_lineno,
                        line=_source(frame.f_code.co_filename, frame.f_lineno),
                        exception_type=type(exc).__name__,
                        exception_value=str(exc),
                    )
                )
                return
            traceback = traceback.tb_next

    def _full(self) -> bool:
        return self._max_spans is not None and len(self._spans) >= self._max_spans

    def _add(self, span: Span) -> None:
        if self._full():
            return
        self._spans.append(span)
        if self._full():
            self._live = False
            self._pending.clear()
            self._disarm()

    def _flush_frame(self, frame: types.FrameType) -> None:
        pending = self._pending.pop(id(frame), None)
        if pending is not None:
            self._emit(pending, snapshot=True)

    def _flush_pending(self) -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        live = _stack_ids()
        for item in pending:
            self._emit(item, snapshot=id(item.frame) in live)

    def _emit(self, pending: _PendingLine, *, snapshot: bool) -> None:
        assignments: dict[str, str] = {}
        if snapshot:
            try:
                bound = self._snapshot_frame(pending.frame)
                names = assigned_names(pending.frame.f_code, pending.line_number)
                assignments = {name: bound[name] for name in names if name in bound}
            except Exception:
                assignments = {}
        self._add(
            LineSpan(
                type="line",
                line_number=pending.line_number,
                line=pending.line,
                assignments=assignments,
            )
        )

    def _snapshot_frame(self, frame: types.FrameType) -> dict[str, str]:
        bound = self._snapshot(frame.f_globals)
        bound.update(self._snapshot(frame.f_locals))
        return bound

    def _snapshot(self, namespace: Any) -> dict[str, str]:
        self._busy = True
        try:
            items = list(namespace.items())
            out: dict[str, str] = {}
            for name, value in items:
                if name in _IGNORE_NAMES or name.startswith("__") or _internal(value):
                    continue
                out[name] = format_value(value)
            return out
        except Exception:
            return {}
        finally:
            self._busy = False

    def _format(self, value: Any) -> str:
        self._busy = True
        try:
            return format_value(value)
        finally:
            self._busy = False


@dataclass
class _PendingLine:
    frame: types.FrameType
    line_number: int
    line: str


class _Capture:
    """Thread-local stdout/stderr capture that leaves IPython's stream alone."""

    def __init__(self, original: Any, buf: io.StringIO, thread_id: int) -> None:
        self._original = original
        self._buf = buf
        self._thread_id = thread_id

    def write(self, data: str) -> int:
        if threading.get_ident() == self._thread_id:
            return self._buf.write(data)
        return self._original.write(data)

    def flush(self) -> None:
        if threading.get_ident() == self._thread_id:
            self._buf.flush()
        else:
            self._original.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _acquire_tool_id() -> int:
    for tool_id in _TOOL_IDS:
        try:
            sys.monitoring.use_tool_id(tool_id, _TOOL_NAME)
            return tool_id
        except ValueError:
            continue
    raise RuntimeError("no free sys.monitoring tool id")


def _stack_ids() -> set[int]:
    ids: set[int] = set()
    frame = sys._getframe()
    while frame is not None:
        ids.add(id(frame))
        frame = frame.f_back
    return ids


def _frame_for_code(code: types.CodeType) -> types.FrameType | None:
    frame = sys._getframe()
    while frame is not None:
        if frame.f_code is code:
            return frame
        frame = frame.f_back
    return None


@lru_cache(maxsize=2048)
def assigned_names(code: types.CodeType, lineno: int) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for instr in dis.get_instructions(code):
        if _instruction_line(instr) != lineno:
            continue
        for name in _store_targets(instr):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return tuple(names)


def format_value(value: Any, *, depth: int = 0) -> str:
    if _internal(value):
        return type(value).__name__
    if isinstance(value, (int, float, complex, bool)) or value is None:
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if depth >= 2:
        return "..."
    if isinstance(value, dict):
        parts = [
            f"{key!r}: {format_value(item, depth=depth + 1)}"
            for key, item in value.items()
            if not _internal(item)
        ]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple, set, frozenset)):
        left, right = (
            ("[", "]")
            if isinstance(value, list)
            else ("(", ")")
            if isinstance(value, tuple)
            else ("{", "}")
        )
        parts = [
            format_value(item, depth=depth + 1) for item in value if not _internal(item)
        ]
        if isinstance(value, tuple) and len(parts) == 1:
            return f"({parts[0]},)"
        return left + ", ".join(parts) + right
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        fields = {
            name: format_value(item, depth=depth + 1)
            for name, item in attrs.items()
            if name not in _IGNORE_NAMES
            and not name.startswith("__")
            and not _internal(item)
        }
        body = ", ".join(f"{name}={item}" for name, item in fields.items())
        return f"{type(value).__name__}({body})"
    text = repr(value)
    return text[:77] + "..." if len(text) > 80 else text


def _internal(value: Any) -> bool:
    try:
        return isinstance(value, _OMIT_TYPES)
    except Exception:
        return True


def _source(filename: str, lineno: int) -> str:
    return linecache.getline(filename, lineno).strip()


def _prime_linecache(filename: str, source: str) -> None:
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)


def _instruction_line(instr: dis.Instruction) -> int | None:
    line = getattr(instr, "line_number", None)
    if line is not None:
        return line
    starts = instr.starts_line
    return starts if isinstance(starts, int) else None


def _store_targets(instr: dis.Instruction) -> list[str]:
    opname, argval = instr.opname, instr.argval
    if opname in _STORE_OPS:
        return [argval] if isinstance(argval, str) else []
    if opname == "STORE_FAST_STORE_FAST" and isinstance(argval, tuple):
        return [name for name in argval if isinstance(name, str)]
    if opname == "STORE_FAST_LOAD_FAST" and isinstance(argval, tuple) and argval:
        return [argval[0]] if isinstance(argval[0], str) else []
    if (
        opname == "LOAD_FAST_STORE_FAST"
        and isinstance(argval, tuple)
        and len(argval) > 1
    ):
        return [argval[1]] if isinstance(argval[1], str) else []
    return []
