from pymongo import ASCENDING
from pymongo.database import Database


def ensure_indexes(database: Database) -> None:
    database.users.create_index("email", unique=True)
    database.roles.create_index("name", unique=True)
    database.subjects.create_index("code", unique=True)
    database.courses.create_index([("subject_id", ASCENDING), ("code", ASCENDING)], unique=True)
    database.syllabi.create_index([("course_id", ASCENDING), ("version", ASCENDING)], unique=True)
    database.syllabus_topics.create_index("syllabus_id")
    database.study_materials.create_index("subject_id")
    database.document_chunks.create_index([("study_material_id", ASCENDING), ("chunk_index", ASCENDING)], unique=True)
    database.document_chunks.create_index([("metadata.subject_id", ASCENDING), ("embedding_model", ASCENDING)])
    database.question_templates.create_index([("name", ASCENDING), ("version", ASCENDING)], unique=True)
    database.questions.create_index([("review_status", ASCENDING), ("subject_id", ASCENDING)])
    database.question_banks.create_index([("subject_id", ASCENDING), ("name", ASCENDING)], unique=True)
    database.model_papers.create_index([("subject_id", ASCENDING), ("name", ASCENDING)], unique=True)
    database.questions.create_index([("syllabus_topic_id", ASCENDING), ("status", ASCENDING)])
    database.question_options.create_index([("question_id", ASCENDING), ("option_key", ASCENDING)], unique=True)
    database.practice_tests.create_index([("student_id", ASCENDING), ("status", ASCENDING)])
    database.student_answers.create_index([("practice_test_id", ASCENDING), ("question_id", ASCENDING)], unique=True)
    database.question_feedback.create_index([("question_id", ASCENDING), ("user_id", ASCENDING)], unique=True)
    database.question_feedback.create_index([("improvement_area", ASCENDING), ("rating", ASCENDING)])
    database.generation_usage.create_index([("user_id", ASCENDING), ("day", ASCENDING)], unique=True)
