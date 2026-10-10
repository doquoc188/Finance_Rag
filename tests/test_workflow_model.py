"""Contract checks for the workflow's explicit offline model fixture."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from app.graph.nodes import GradeDocuments, GradeHallucination


def test_chat_model_accepts_direct_and_prompt_inputs(workflow_chat_model):
    direct = workflow_chat_model.invoke("Test question")
    chain = ChatPromptTemplate.from_messages([("user", "{question}")]) | workflow_chat_model
    composed = chain.invoke({"question": "Test question"}, config={"tags": ["offline-test"]})

    assert isinstance(direct, AIMessage)
    assert isinstance(composed, AIMessage)
    assert direct.content == composed.content == "Offline workflow test response."


@pytest.mark.parametrize("schema", [GradeDocuments, GradeHallucination])
def test_structured_grades_use_the_requested_schema(workflow_chat_model, schema):
    grade = workflow_chat_model.with_structured_output(schema).invoke("Test document")

    assert isinstance(grade, schema)
    assert grade.binary_score == "yes"
    if schema is GradeHallucination:
        assert grade.is_useful == "yes"


def test_structured_grades_do_not_hide_schema_errors(workflow_chat_model):
    class UnsupportedGrade(BaseModel):
        required_evidence: str

    with pytest.raises(ValidationError):
        workflow_chat_model.with_structured_output(UnsupportedGrade).invoke("Test document")
