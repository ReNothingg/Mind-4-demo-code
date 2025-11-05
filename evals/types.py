from dataclasses import dataclass, field
from typing import Any, Literal, overload

Message = dict[str, Any]
MessageList = list[Message]



@dataclass
class SamplerResponse:
    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]

class SamplerBase:
    def __call__(
        self,
        message_list: MessageList,
    ) -> SamplerResponse:
        raise NotImplementedError


@dataclass
class EvalResult:
    score: float | None
    metrics: dict[str, float] | None
    htmls: list[str]
    convos: list[MessageList]
    metadata: dict[str, Any] | None


@dataclass
class SingleEvalResult:
    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None
    example_level_metadata: dict[str, Any] | None = (
        None
    )


class Eval:

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError

