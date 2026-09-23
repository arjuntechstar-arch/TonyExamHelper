# AI Codebase Generation Pack

Put this folder at the root of a new repository or copy it into `docs/`.

Give the coding agent this instruction:

> Read every file in `docs/` before coding. Follow `00-MASTER-CODING-PROMPT.md`. Start Phase 1 only. After implementation, build and test the solution, fix failures, and report files changed, tests run, failures and the next phase.

Do not ask the coding agent to generate the entire application in one unverified step. Use the defined phases.
