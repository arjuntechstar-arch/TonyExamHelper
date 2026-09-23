import mongomock

from app.database import ensure_indexes
from app.models import SubjectDocument
from app.repositories import SubjectRepository


def test_subject_repository_persists_and_queries_subjects() -> None:
    database = mongomock.MongoClient().ai_examination_studio
    repository = SubjectRepository(database)
    subject = repository.add_subject(SubjectDocument(code="CS101", name="Computer Science"))

    assert repository.get(subject.id) is not None
    assert repository.get_by_code("CS101").name == "Computer Science"
    assert [item["code"] for item in repository.list()] == ["CS101"]


def test_index_bootstrap_enforces_subject_codes() -> None:
    database = mongomock.MongoClient().ai_examination_studio
    ensure_indexes(database)
    database.subjects.insert_one({"_id": "first", "code": "CS101"})

    try:
        database.subjects.insert_one({"_id": "second", "code": "CS101"})
    except Exception as error:
        assert error.__class__.__name__ == "DuplicateKeyError"
    else:
        raise AssertionError("Expected the unique subject code index to reject a duplicate.")
