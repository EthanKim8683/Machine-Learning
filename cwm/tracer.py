"""Execution tracer that records call, line, return, and exception spans."""

from __future__ import annotations

import dis
import io
import linecache
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict

# BaseSpan.type is annotated as types.FrameType in the contract; pydantic cannot
# build a schema for frame objects unless arbitrary types are allowed.
_base_model_config = BaseModel.model_config
BaseModel.model_config = ConfigDict(arbitrary_types_allowed=True)
from tracer_contract import (  # noqa: E402
    CallSpan,
    ExceptionSpan,
    LineSpan,
    ReturnSpan,
    Trace as ContractTrace,
    Tracer as TracerContract,
)

BaseModel.model_config = _base_model_config

__all__ = [
    "CallSpan",
    "ExceptionSpan",
    "LineSpan",
    "ReturnSpan",
    "Trace",
    "Tracer",
]

TRACER_FILE = os.path.abspath(__file__)

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
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
        "__annotations__",
        "__stdout__",
        "__exception__",
        "__return__",
        "__module__",
        "__class__",
        "__dict__",
        "__weakref__",
    }
)

Span = CallSpan | LineSpan | ReturnSpan | ExceptionSpan


class Trace(ContractTrace):
    spans: list[Span]


class Tracer(TracerContract):
    """Record observation-action spans while a ``with tracer.trace():`` block runs."""

    def __init__(self) -> None:
        self._traces: list[Trace] = []
        self._events: list[_Event] = []
        self._pending: dict[int, _PendingLine] = {}
        self._attached: list[types.FrameType] = []
        self._rendering = False

    @contextmanager
    def trace(self) -> Iterator[Tracer]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        old_hook = sys.gettrace()
        self._events = []
        self._pending = {}
        self._rendering = False
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            self._install()
            yield self
        finally:
            self._uninstall(old_hook)
            self._flush_all()
            sys.stdout = old_out
            sys.stderr = old_err
            self._traces.append(
                Trace(
                    spans=self._to_spans(),
                    stdout=stdout.getvalue(),
                    stderr=stderr.getvalue(),
                )
            )

    def traces(self) -> list[Trace]:
        return list(self._traces)

    def _callback(self, frame: types.FrameType, event: str, arg: Any) -> Any:
        if self._rendering or not self._should_trace(frame):
            return None
        if event == "call":
            self._on_call(frame)
        elif event == "line":
            self._on_line(frame)
        elif event == "return":
            self._on_return(frame, arg)
        elif event == "exception":
            self._on_exception(frame, arg)
        return self._callback

    def _install(self) -> None:
        self._attached = []
        sys.settrace(self._callback)
        frame = sys._getframe().f_back
        while frame is not None:
            if self._should_trace(frame):
                frame.f_trace = self._callback
                self._attached.append(frame)
            frame = frame.f_back

    def _uninstall(self, old_hook: Any) -> None:
        sys.settrace(old_hook)
        for frame in self._attached:
            if frame.f_trace is self._callback:
                frame.f_trace = old_hook
        self._attached = []

    def _should_trace(self, frame: types.FrameType) -> bool:
        filename = frame.f_code.co_filename
        if filename.startswith("<frozen "):
            return False
        try:
            path = os.path.abspath(filename).replace("\\", "/")
        except (OSError, ValueError):
            path = filename.replace("\\", "/")
        if path == TRACER_FILE:
            return False
        for marker in _SKIP_MARKERS:
            if marker in path:
                return False
        if not filename.startswith("<"):
            for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix):
                if prefix and path.startswith(
                    os.path.abspath(prefix).replace("\\", "/")
                ):
                    return False
        return True

    def _on_call(self, frame: types.FrameType) -> None:
        if _is_module(frame):
            return
        caller = frame.f_back
        if caller is not None:
            self._flush_frame(caller)
        self._events.append(
            _Event(
                kind="call",
                line_number=frame.f_code.co_firstlineno,
                line=_source_at(frame.f_code.co_filename, frame.f_code.co_firstlineno),
                arguments=self._snapshot(frame.f_locals),
            )
        )

    def _on_line(self, frame: types.FrameType) -> None:
        self._flush_frame(frame)
        self._pending[id(frame)] = _PendingLine(
            frame=frame,
            line_number=frame.f_lineno,
            line=_source_line(frame),
        )

    def _on_return(self, frame: types.FrameType, retval: Any) -> None:
        self._flush_frame(frame)
        if _is_module(frame):
            return
        self._events.append(
            _Event(
                kind="return",
                line_number=frame.f_lineno,
                line=_source_line(frame),
                return_value=self._format(retval),
            )
        )

    def _on_exception(self, frame: types.FrameType, arg: Any) -> None:
        self._flush_frame(frame)
        exc_type, exc_value, _tb = arg
        self._events.append(
            _Event(
                kind="exception",
                line_number=frame.f_lineno,
                line=_source_line(frame),
                exception_type=getattr(exc_type, "__name__", str(exc_type)),
                exception_value=str(exc_value),
            )
        )

    def _flush_frame(self, frame: types.FrameType) -> None:
        pending = self._pending.pop(id(frame), None)
        if pending is not None:
            self._emit_line(pending)

    def _flush_all(self) -> None:
        for pending in list(self._pending.values()):
            self._emit_line(pending)
        self._pending.clear()

    def _emit_line(self, pending: _PendingLine) -> None:
        current = self._snapshot_frame(pending.frame)
        assigned = assigned_names(pending.frame.f_code, pending.line_number)
        self._events.append(
            _Event(
                kind="line",
                line_number=pending.line_number,
                line=pending.line,
                assignments={
                    name: current[name] for name in assigned if name in current
                },
            )
        )

    def _to_spans(self) -> list[Span]:
        spans: list[Span] = []
        for event in self._events:
            if event.kind == "call":
                spans.append(
                    CallSpan(
                        type="call",
                        line_number=event.line_number,
                        line=event.line,
                        arguments=event.arguments,
                    )
                )
            elif event.kind == "line":
                spans.append(
                    LineSpan(
                        type="line",
                        line_number=event.line_number,
                        line=event.line,
                        assignments=event.assignments,
                    )
                )
            elif event.kind == "return":
                spans.append(
                    ReturnSpan(
                        type="return",
                        line_number=event.line_number,
                        line=event.line,
                        return_value=event.return_value,
                    )
                )
            else:
                spans.append(
                    ExceptionSpan(
                        type="exception",
                        line_number=event.line_number,
                        line=event.line,
                        exception_type=event.exception_type,
                        exception_value=event.exception_value,
                    )
                )
        return spans

    def _snapshot_frame(self, frame: types.FrameType) -> dict[str, str]:
        bound = self._snapshot(frame.f_globals)
        bound.update(self._snapshot(frame.f_locals))
        return bound

    def _snapshot(self, namespace: dict[str, Any]) -> dict[str, str]:
        self._rendering = True
        try:
            out: dict[str, str] = {}
            for name, value in namespace.items():
                if name in _IGNORE_NAMES or name.startswith("__"):
                    continue
                if _is_internal(value):
                    continue
                out[name] = format_value(value)
            return out
        finally:
            self._rendering = False

    def _format(self, value: Any) -> str:
        self._rendering = True
        try:
            return format_value(value)
        finally:
            self._rendering = False


@dataclass
class _PendingLine:
    frame: types.FrameType
    line_number: int
    line: str


@dataclass
class _Event:
    kind: Literal["call", "line", "return", "exception"]
    line_number: int
    line: str
    arguments: dict[str, str] = field(default_factory=dict)
    assignments: dict[str, str] = field(default_factory=dict)
    return_value: str = "None"
    exception_type: str = ""
    exception_value: str = ""


def assigned_names(code: types.CodeType, lineno: int) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for instr in dis.get_instructions(code):
        if _instruction_line(instr) != lineno:
            continue
        for name in _store_targets(instr):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def format_value(value: Any, *, depth: int = 0) -> str:
    if _is_internal(value):
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
            if not _is_internal(item)
        ]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple, set, frozenset)):
        brackets = (
            ("[", "]")
            if isinstance(value, list)
            else ("(", ")")
            if isinstance(value, tuple)
            else ("{", "}")
        )
        parts = [
            format_value(item, depth=depth + 1)
            for item in value
            if not _is_internal(item)
        ]
        if isinstance(value, tuple) and len(parts) == 1:
            return f"({parts[0]},)"
        return brackets[0] + ", ".join(parts) + brackets[1]
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        fields = {
            name: format_value(item, depth=depth + 1)
            for name, item in attrs.items()
            if name not in _IGNORE_NAMES
            and not name.startswith("__")
            and not _is_internal(item)
        }
        body = ", ".join(f"{name}={item}" for name, item in fields.items())
        return f"{type(value).__name__}({body})"
    text = repr(value)
    return text[:77] + "..." if len(text) > 80 else text


def _is_internal(value: Any) -> bool:
    return isinstance(value, _OMIT_TYPES)


def _is_module(frame: types.FrameType) -> bool:
    return frame.f_code.co_name == "<module>"


def _source_line(frame: types.FrameType) -> str:
    return _source_at(frame.f_code.co_filename, frame.f_lineno)


def _source_at(filename: str, lineno: int) -> str:
    return linecache.getline(filename, lineno).strip()


def _instruction_line(instr: dis.Instruction) -> int | None:
    line = getattr(instr, "line_number", None)
    if line is not None:
        return line
    starts = instr.starts_line
    return starts if isinstance(starts, int) else None


def _store_targets(instr: dis.Instruction) -> list[str]:
    opname = instr.opname
    argval = instr.argval
    if opname in {
        "STORE_FAST",
        "STORE_NAME",
        "STORE_GLOBAL",
        "STORE_DEREF",
        "STORE_FAST_MAYBE_NULL",
    }:
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
