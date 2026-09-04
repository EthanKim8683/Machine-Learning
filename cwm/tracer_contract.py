from abc import ABC, abstractmethod
from contextlib import contextmanager
from pydantic import BaseModel
import types
from typing import Literal, Optional


class BaseSpan(BaseModel):
    type: types.FrameType
    line_number: int
    line: str


class CallSpan(BaseSpan):
    type: Literal["call"]
    arguments: dict[str, str]


class ReturnSpan(BaseSpan):
    type: Literal["return"]
    return_value: str


class LineSpan(BaseSpan):
    type: Literal["line"]
    assignments: dict[str, str]


class ExceptionSpan(BaseSpan):
    type: Literal["exception"]
    exception_type: str
    exception_value: str


class Trace(BaseModel):
    spans: list[BaseSpan]
    stdout: str
    stderr: str


class Tracer(ABC):
    @abstractmethod
    def __init__(self, max_spans: Optional[int] = None):
        pass

    @abstractmethod
    @contextmanager
    def trace(self):
        pass

    @abstractmethod
    def trace_str(self, src: str):
        pass

    @abstractmethod
    def traces(self) -> list[Trace]:
        pass

    @abstractmethod
    def clear(self):
        pass
