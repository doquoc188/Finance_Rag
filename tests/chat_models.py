"""Reusable local chat model; never imported by production source."""

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class ModelCall:
    role: str
    input: str
    schema: type[BaseModel] | None = None


class WorkflowChatModel(FakeListChatModel):
    """Runnable chat interface with AIMessage responses and schema-validated grades."""

    role: str = "test"
    binary_score: str = "yes"
    binary_scores: tuple[str, ...] = ()
    calls: list[ModelCall] = Field(default_factory=list)

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(ModelCall(self.role, "\n".join(str(m.content) for m in messages)))
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)

    def with_structured_output(
        self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        # Any matches LangChain's extensible schema/options boundary, not model results.
        if include_raw or not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise NotImplementedError("The test model supports Pydantic schemas only")

        def grade(value: LanguageModelInput) -> dict[str, Any] | BaseModel:
            messages = self._convert_input(value).to_messages()
            score_index = sum(call.schema is not None for call in self.calls)
            self.calls.append(
                ModelCall(
                    self.role, "\n".join(str(message.content) for message in messages), schema
                )
            )
            score = (
                self.binary_scores[score_index]
                if score_index < len(self.binary_scores)
                else self.binary_score
            )
            return schema.model_validate({"binary_score": score, "is_useful": "yes"})

        return RunnableLambda(grade)
