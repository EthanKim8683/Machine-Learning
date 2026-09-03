# Assignments inside traced blocks exist to produce spans, not to be read.
# ruff: noqa: F841

import sys

import pytest

from tracer import CallSpan, ExceptionSpan, LineSpan, ReturnSpan, Tracer
from tracer_contract import Tracer as TracerContract


def spans_of(trace, type: str):
    return [span for span in trace.spans if span.type == type]


def by_line(trace, line: str, type: str = "line"):
    return [span for span in trace.spans if span.line == line and span.type == type]


def test_is_contract_tracer() -> None:
    assert issubclass(Tracer, TracerContract)
    assert Tracer().traces() == []


def test_assignment_shows_locals_after_the_line() -> None:
    tracer = Tracer()
    with tracer.trace():
        x = 1
        y = x + 2
    [trace] = tracer.traces()
    lines = spans_of(trace, "line")
    assert lines[0].line == "x = 1"
    assert lines[0].assignments == {"x": "1"}
    assert lines[1].line == "y = x + 2"
    assert lines[1].assignments == {"y": "3"}
    assert isinstance(lines[0], LineSpan)


def test_function_call_and_return() -> None:
    def add(a, b):
        return a + b

    tracer = Tracer()
    with tracer.trace():
        add(2, 3)
    [trace] = tracer.traces()
    calls = spans_of(trace, "call")
    returns = spans_of(trace, "return")
    assert calls[0].arguments == {"a": "2", "b": "3"}
    assert calls[0].line.startswith("def add")
    assert returns[0].return_value == "5"
    assert isinstance(calls[0], CallSpan)
    assert isinstance(returns[0], ReturnSpan)


def test_call_site_is_recorded_before_the_callee_runs() -> None:
    def foo():
        bar()
        x = 1

    def bar():
        y = 2

    tracer = Tracer()
    with tracer.trace():
        foo()
    [trace] = tracer.traces()

    def index(*, type: str, line: str = "", startswith: str = "") -> int:
        for i, span in enumerate(trace.spans):
            if span.type != type:
                continue
            if line and span.line != line:
                continue
            if startswith and not span.line.startswith(startswith):
                continue
            return i
        raise AssertionError(f"missing {type} {line}{startswith}")

    assert index(type="line", line="foo()") < index(type="call", startswith="def foo")
    assert index(type="call", startswith="def foo") < index(type="line", line="bar()")
    assert index(type="line", line="bar()") < index(type="call", startswith="def bar")
    assert index(type="call", startswith="def bar") < index(type="line", line="y = 2")
    assert index(type="return", line="y = 2") < index(type="line", line="x = 1")


def test_nested_calls_include_each_def_line() -> None:
    def foo():
        return bar()

    def bar():
        return 7

    tracer = Tracer()
    with tracer.trace():
        foo()
    [trace] = tracer.traces()
    assert any(
        span.type == "call" and span.line.startswith("def foo") for span in trace.spans
    )
    assert any(
        span.type == "call" and span.line.startswith("def bar") for span in trace.spans
    )
    assert any(
        span.type == "return" and span.return_value == "7" and span.line == "return 7"
        for span in trace.spans
    )
    assert any(
        span.type == "return"
        and span.return_value == "7"
        and span.line == "return bar()"
        for span in trace.spans
    )


def test_reassignment_is_logged_even_when_value_is_unchanged() -> None:
    tracer = Tracer()
    with tracer.trace():
        x = 1
        x = 1
        xs = [1]
        xs.append(2)
    [trace] = tracer.traces()
    x_lines = [span.assignments for span in by_line(trace, "x = 1")]
    assert x_lines == [{"x": "1"}, {"x": "1"}]
    assert by_line(trace, "xs = [1]")[0].assignments == {"xs": "[1]"}
    assert by_line(trace, "xs.append(2)")[0].assignments == {}


def test_global_reassignment_is_logged() -> None:
    count = 0

    def bump():
        nonlocal count
        count = 0
        count += 1

    tracer = Tracer()
    with tracer.trace():
        bump()
    [trace] = tracer.traces()
    body = {span.line: span.assignments for span in spans_of(trace, "line")}
    assert body["count = 0"] == {"count": "0"}
    assert body["count += 1"] == {"count": "1"}


def test_only_assigned_names_are_shown() -> None:
    def foo(n):
        total = 0
        total += n
        return total

    tracer = Tracer()
    with tracer.trace():
        foo(3)
    [trace] = tracer.traces()
    body = {span.line: span.assignments for span in spans_of(trace, "line")}
    assert body["total = 0"] == {"total": "0"}
    assert "n" not in body["total = 0"]
    assert body["total += n"] == {"total": "3"}
    assert body["return total"] == {}

    def loop(n):
        for i in range(n):
            pass

    tracer = Tracer()
    with tracer.trace():
        loop(2)
    [trace] = tracer.traces()
    for_lines = [span.assignments for span in by_line(trace, "for i in range(n):")]
    assert for_lines[0] == {"i": "0"}
    assert for_lines[1] == {"i": "1"}
    assert all(span.keys() == {"i"} for span in for_lines)


def test_uncaught_exception_is_recorded() -> None:
    tracer = Tracer()
    with pytest.raises(ZeroDivisionError):
        with tracer.trace():
            1 / 0
    [trace] = tracer.traces()
    exceptions = spans_of(trace, "exception")
    assert exceptions
    assert exceptions[0].exception_type == "ZeroDivisionError"
    assert "division" in exceptions[0].exception_value.lower()
    assert isinstance(exceptions[0], ExceptionSpan)


def test_context_manager_hides_tracer_and_contextlib_internals() -> None:
    def foo() -> int:
        return bar()

    def bar() -> int:
        return 1

    tracer = Tracer()
    with tracer.trace():
        foo()
    [trace] = tracer.traces()
    text = " ".join(span.line for span in trace.spans)
    assert any(span.type == "call" for span in trace.spans)
    assert any(
        span.type == "return" and span.return_value == "1" for span in trace.spans
    )
    assert "__exit__" not in text
    assert "__enter__" not in text
    assert "contextlib" not in text


def test_locals_omit_modules() -> None:
    tracer = Tracer()
    with tracer.trace():
        import sys as imported_sys

        x = 1
    assert imported_sys is sys
    [trace] = tracer.traces()
    for span in spans_of(trace, "line"):
        assert "imported_sys" not in span.assignments
        assert "sys" not in span.assignments
    assert by_line(trace, "x = 1")[0].assignments == {"x": "1"}


def test_locals_omit_functions() -> None:
    tracer = Tracer()
    with tracer.trace():

        def foo():
            return 1

        foo()
    [trace] = tracer.traces()
    for span in spans_of(trace, "line"):
        assert "foo" not in span.assignments


def test_stdlib_and_builtin_calls_are_not_traced() -> None:
    tracer = Tracer()
    with tracer.trace():
        xs = [3, 1, 2]
        ys = sorted(xs)
        n = len(ys)
    [trace] = tracer.traces()
    assert spans_of(trace, "call") == []
    assert by_line(trace, "ys = sorted(xs)")[0].assignments == {"ys": "[1, 2, 3]"}
    assert by_line(trace, "n = len(ys)")[0].assignments == {"n": "3"}


def test_stdout_and_stderr_are_captured(capsys: pytest.CaptureFixture[str]) -> None:
    tracer = Tracer()
    with tracer.trace():
        print("hello")
        print("world", file=sys.stderr)
    leaked = capsys.readouterr()
    assert "hello" not in leaked.out
    assert "world" not in leaked.err
    [trace] = tracer.traces()
    assert trace.stdout == "hello\n"
    assert trace.stderr == "world\n"
    assert all(span.type != "output" for span in trace.spans)
    assert by_line(trace, 'print("hello")')


def test_multiple_trace_sessions() -> None:
    tracer = Tracer()
    with tracer.trace():
        a = 1
    with tracer.trace():
        b = 2
    traces = tracer.traces()
    assert len(traces) == 2
    assert by_line(traces[0], "a = 1")[0].assignments == {"a": "1"}
    assert by_line(traces[1], "b = 2")[0].assignments == {"b": "2"}
    assert traces[0] is not traces[1]


def test_values_are_repr_strings() -> None:
    def greet(name):
        msg = "hi " + name
        return msg

    tracer = Tracer()
    with tracer.trace():
        greet("ada")
    [trace] = tracer.traces()
    assert any(
        span.type == "call" and span.arguments == {"name": "'ada'"}
        for span in trace.spans
    )
    assert any(
        span.type == "line" and span.assignments == {"msg": "'hi ada'"}
        for span in trace.spans
    )
    assert any(
        span.type == "return" and span.return_value == "'hi ada'"
        for span in trace.spans
    )


def test_lambda_call_and_return() -> None:
    f = lambda n: n + 1  # noqa: E731

    tracer = Tracer()
    with tracer.trace():
        f(10)
    [trace] = tracer.traces()
    assert spans_of(trace, "call")
    assert spans_of(trace, "return")[0].return_value == "11"


def test_user_instance_values_are_compact() -> None:
    class Counter:
        def __init__(self):
            self.n = 0

        def bump(self):
            self.n += 1
            return self.n

    tracer = Tracer()
    with tracer.trace():
        c = Counter()
        c.bump()
    [trace] = tracer.traces()
    bump = next(span for span in spans_of(trace, "return") if span.return_value == "1")
    assert bump.return_value == "1"
    assert any(
        span.type == "call" and span.line.startswith("def __init__")
        for span in trace.spans
    )
    values = []
    for span in trace.spans:
        if span.type == "call":
            values.extend(span.arguments.values())
        elif span.type == "line":
            values.extend(span.assignments.values())
    dumped = " ".join(values)
    assert "Counter(" in dumped
    assert "__dict__" not in dumped


def test_line_numbers_are_positive_and_source_is_stripped() -> None:
    tracer = Tracer()
    with tracer.trace():
        x = 1
    [trace] = tracer.traces()
    for span in trace.spans:
        assert span.line_number >= 1
        assert span.line == span.line.strip()
