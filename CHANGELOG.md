# Changelog

## Unreleased

### Phase 15

- Completed the README, API contract, environment, and deployment guidance.
- Added non-secret Docker Compose configuration, a Docker build context
  exclusion file, and GitHub Actions validation for backend and frontend.

### Phase 4

- Added faculty/admin-protected subject, course, syllabus, and syllabus-topic API endpoints.

### Phase 3

- Added bcrypt password hashing, JWT authentication, and Student/Faculty/Admin role enforcement.
- Added `/api/auth/login` and `/api/auth/me` contracts plus authentication service tests.

### Phase 2

- Replaced the relational persistence layer with MongoDB configuration, document models, collection indexes, and repositories.
- Added MongoDB repository and index tests.

### Phase 1

- Added Angular and FastAPI application shells.
- Added environment-backed backend settings, JSON logging, request IDs, and `/api/health`.
- Added backend and frontend foundation tests.
- Added a development API proxy, request-ID validation, and a consistent backend error contract.
