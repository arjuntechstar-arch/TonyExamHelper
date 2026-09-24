import mongomock
import pytest
from fastapi import HTTPException

from app.api.templates import TemplatePayload, TemplateUpdate, create_template, list_templates, update_template
from app.database import ensure_indexes


def payload(name: str = "Direct Concept", version: str = "1.0") -> TemplatePayload:
    return TemplatePayload(
        name=name,
        question_type="MCQ",
        pattern="Direct Concept",
        required_fields=["question_text", "options", "correct_answer"],
        supported_difficulties=["Easy", "Medium", "Hard"],
        supported_bloom_levels=["Remember", "Understand", "Apply"],
        version=version,
    )


@pytest.fixture
def database():
    database = mongomock.MongoClient().test
    ensure_indexes(database)
    return database


def test_template_crud_and_filter_selection(database) -> None:
    created = create_template(payload(), database, object())

    selected = list_templates(question_type="MCQ", difficulty="Medium", bloom_level="Apply", database=database, _=object())
    updated = update_template(created.id, TemplateUpdate(pattern="Scenario Based"), database, object())

    assert [item.id for item in selected] == [created.id]
    assert updated.pattern == "Scenario Based"


def test_template_name_and_version_must_be_unique(database) -> None:
    create_template(payload(), database, object())

    with pytest.raises(HTTPException) as error:
        create_template(payload(), database, object())

    assert error.value.status_code == 409


def test_template_payload_requires_supported_configuration() -> None:
    with pytest.raises(ValueError):
        TemplatePayload(
            name="Invalid",
            question_type="MCQ",
            pattern="Direct Concept",
            required_fields=[],
            supported_difficulties=["Medium"],
            supported_bloom_levels=["Apply"],
            version="1.0",
        )