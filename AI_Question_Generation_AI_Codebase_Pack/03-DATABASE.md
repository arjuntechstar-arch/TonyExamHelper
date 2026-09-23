# Database Design

Core SQL Server entities:

Users
Roles
UserRoles
Subjects
Courses
Syllabus
SyllabusTopics
StudyMaterials
DocumentChunks
QuestionTemplates
ExamPatterns
Questions
QuestionOptions
QuestionValidation
QuestionSimilarity
QuestionBanks
PracticeTests
PracticeQuestions
StudentAnswers
PerformanceAnalytics

Relationships:
Subject → Course → Syllabus → SyllabusTopics → Questions
Subject → StudyMaterials → DocumentChunks
QuestionBank → Questions
PracticeTest → PracticeQuestions → Questions
PracticeTest → StudentAnswers

Required audit fields where appropriate:
CreatedAt, UpdatedAt, CreatedBy, UpdatedBy, Status.

Use migrations, foreign keys, indexes and appropriate SQL Server types.
