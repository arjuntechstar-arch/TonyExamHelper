# API Contracts

Base path: `/api`

POST /auth/login
GET /auth/me

POST /subjects
GET /subjects
POST /syllabus
GET /syllabus/{id}
POST /syllabus/{id}/topics

POST /materials/upload
POST /materials/{id}/process
GET /materials/{id}/status

POST /retrieval/materials/{id}/index
POST /retrieval/search

GET /templates
POST /templates
PUT /templates/{id}

POST /questions/generate
POST /questions/generate/batch
POST /questions
GET /questions/review
POST /questions/{id}/validate
POST /questions/{id}/approve
POST /questions/{id}/reject

GET /question-bank
POST /question-bank
POST /question-bank/{id}/approve

POST /model-papers
GET /model-papers/{id}
POST /model-papers/{id}/publish

POST /practice/start
POST /practice/{id}/submit
GET /practice/{id}/result

GET /analytics/student
GET /analytics/student/topics
GET /analytics/student/difficulty

All endpoints require request validation, RBAC where applicable, consistent error responses and request/correlation IDs.
