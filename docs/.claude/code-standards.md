## Coding Standards
All new code and modifications to existing code must follow these standards. Prioritize correctness, readability, maintainability, reliability, and simplicity.

1. General Design Principles
Follow these principles:
- SOLID: Apply object-oriented design principles where they improve maintainability.
- DRY: Avoid unnecessary code duplication.
- KISS: Prefer simple solutions over complex ones.
- YAGNI: Do not implement features that are not currently required.
- Composition over inheritance: Prefer combining focused components over complex class hierarchies.
- Separation of concerns: Keep unrelated responsibilities separate.
- Dependency inversion: Depend on stable abstractions rather than concrete implementations.

Additional requirements:
- Prefer readable code over clever code.
- Make dependencies explicit.
- Keep side effects isolated.
- Avoid hidden behavior and implicit dependencies.
- Avoid premature optimization.
- Avoid unnecessary abstractions and frameworks.

2. Architecture and Separation of Concerns
Use clear architectural boundaries. Prefer Hexagonal Architecture (Ports and Adapters) principles where they provide practical value. Do not enforce unnecessary layers or abstractions.

Responsibilities:
- Workers handle message consumption and processing orchestration.
- Services contain reusable application and business logic.
- Repositories or infrastructure adapters handle external resources.
- Model wrappers isolate model loading and inference.
- Configuration modules handle environment-specific settings.
- Dependency direction: Worker → Application Service → Domain Logic
Infrastructure adapters implement the interfaces required by application logic.

Additional requirements:
- Keep business logic independent of infrastructure where practical.
- Keep Redis communication separate from processing logic.
- Keep database operations separate from business rules.
- Keep model inference separate from orchestration.
- Avoid circular imports.
- Avoid classes and modules with unrelated responsibilities.
- Do not introduce architectural layers without a clear need.

3. Python Coding Standards
- Follow PEP 8.
- Add type hints to public functions, methods, and important internal interfaces.
- Use descriptive names.
- Prefer explicit arguments over complex positional arguments.
- Use pathlib for filesystem paths.
- Use context managers for resources that require cleanup.
- Use structured logging instead of print().
- Use specific exception types.
- Avoid bare except clauses.
- Avoid mutable default arguments.
- Avoid wildcard imports.
- Avoid unnecessary global state.
- Use constants for shared configuration values.
- Organize imports consistently.
- Prefer standard library solutions when sufficient.
- Use dataclasses or typed structures for important data contracts.
- Keep functions focused on one responsibility.
- Keep classes cohesive.

4. NASA/JPL Power of 10
Apply the safety and verification principles of Gerard J. Holzmann's NASA/JPL Power of 10 to all code affecting data integrity, processing reliability, persistence, and recovery. The original rules target C — adapt language-specific restrictions to Python while preserving their goals.

Rule 1 — Use simple control flow
- Avoid deeply nested conditionals.
- Avoid complex branching.
- Avoid recursion in critical processing paths.
- Prefer explicit loops and straightforward execution paths.
- Make failure paths easy to identify.
- Avoid hidden control flow and implicit behavior.

Rule 2 — Bound loops and operations
- Every processing loop must have a clear termination condition.
- Set explicit limits for retries, batch sizes, and pagination.
- Prevent infinite retry loops.
- Use timeouts for blocking operations.
- Support controlled shutdown in long-running workers.

Rule 3 — Control memory allocation and growth
- Avoid unbounded queues, caches, and collections.
- Set limits for batches and in-memory buffers.
- Use streaming or chunked processing for large datasets.
- Reuse expensive resources (model instances, connection pools).
- Avoid loading unnecessary copies of large models.
- Release resources reliably.

Rule 4 — Keep functions small
- Keep functions focused on one responsibility.
- Use approximately 60 lines of executable code as a maximum guideline.
- Split complex functions into smaller, testable operations.
- Avoid excessive parameters and hidden dependencies.
- Do not introduce unnecessary abstractions just to satisfy a line limit.

Rule 5 — Use explicit assertions and validation
- Validate external inputs at system boundaries.
- Check preconditions before critical operations.
- Check postconditions after important state changes.
- Use assertions for internal invariants.
- Never use assertions as a replacement for input validation.
- Never rely on assertions for security checks or required production behavior.
- Make invalid states explicit and fail clearly.

Rule 6 — Minimize variable scope
- Declare variables close to their first use.
- Avoid unnecessary module-level mutable state.
- Avoid shared mutable state between concurrent tasks.
- Prefer immutable data where practical.
- Keep object lifetimes limited.
- Avoid variables whose meaning changes during execution.

Rule 7 — Check operation results
- Check return values from critical operations.
- Handle database, Redis, and storage failures explicitly.
- Validate model outputs before persistence.
- Never silently ignore failed operations.
- Never acknowledge messages before successful processing.

Rule 8 — Limit dynamic behavior
- Avoid unnecessary metaprogramming.
- Do not use eval() or exec().
- Avoid runtime modification of classes and modules.
- Avoid excessive reflection and dynamic imports.
- Prefer explicit configuration and dependency injection.
- Make dependencies and execution paths easy to inspect.

Rule 9 — Keep interfaces simple
- Prefer simple, explicit data structures.
- Avoid deeply nested object references.
- Avoid unnecessary mutable object sharing.
- Avoid complex callback chains.
- Keep public interfaces small and predictable.
- Use typed structures for important message contracts.
- Avoid exposing internal implementation details.

Rule 10 — Verify code continuously
- Run automated tests after meaningful changes.
- Use static analysis and linting.
- Use type checking for important interfaces.
- Fix warnings rather than suppressing them without justification.
- Test failure paths and edge cases.
- Verify critical behavior through tests, not assumptions.
These principles improve reliability but do not constitute formal safety certification.

5. Error Handling and Reliability
- Handle errors at the layer that has enough context to respond correctly.
- Never silently ignore exceptions.
- Avoid exceptions for normal control flow.
- Preserve the original exception when raising a new one.
- Include useful context in error messages.
- Never expose secrets in exceptions or logs.
- Distinguish transient failures from permanent failures.
- Retry only operations that are safe to repeat.
- Use bounded retries with backoff.
- Avoid infinite retry loops.
- Release resources after failures.
- Preserve acknowledgment and dead-letter guarantees.

Distributed Processing
Assume messages can be delivered more than once.
- Design processing operations to be idempotent where possible.
- Never assume exactly-once message delivery.
- Make partial failures recoverable.
- Use transactions for related database changes.
- Preserve message acknowledgment semantics.
- Define behavior for duplicate messages.
- Make state transitions explicit.

6. Data Validation and Integrity
- Validate all external inputs at system boundaries.
- Do not assume Redis messages contain valid data.
- Validate required fields and expected types.
- Fail clearly when message contracts are invalid.
- Avoid silently replacing invalid values with arbitrary defaults.
- Keep serialization and deserialization explicit.
- Use consistent timestamp and timezone handling.
- Use appropriate numeric types for precision-sensitive values.
- Avoid unnecessary copies of large datasets.
- Preserve data provenance where relevant.

7. Logging and Observability
Use Python's standard logging module or the project's established logging solution.
Level | Purpose
DEBUG | Diagnostic information
INFO  | Important operational events
WARNING | Unexpected but recoverable conditions
ERROR | Failed operations
CRITICAL | Severe failures

- Log worker startup and shutdown.
- Log processing success and failure.
- Log retries and dead-letter events.
- Include task IDs and correlation IDs where available.
- Record processing durations where useful.
- Avoid logging credentials, tokens, and sensitive payloads.
- Avoid excessive logging in high-frequency loops.
- Prefer structured and searchable messages.

8. Concurrency and Resource Management
- Make concurrency decisions explicit.
- Do not share mutable state between concurrent tasks without synchronization.
- Avoid blocking asynchronous execution with synchronous operations.
- Use connection pools where appropriate.
- Define timeouts for external operations.
- Limit concurrent model inference according to available resources.
- Release database connections, files, and other resources reliably.
- Avoid loading multiple copies of large models unnecessarily.
- Consider CPU, GPU, memory, and network constraints.

9. Performance and Scalability
- Optimize based on measured bottlenecks.
- Do not sacrifice readability for speculative performance improvements.
- Avoid N+1 database queries.
- Use batch operations where appropriate.
- Avoid loading entire datasets into memory unnecessarily.
- Use streaming or chunked processing for large datasets.
- Reuse expensive resources.
- Define limits for batch sizes and concurrent operations.
- Measure performance before and after significant optimizations.

10. Security
- Never hardcode credentials or secrets.
- Read sensitive configuration from approved environment variables or secret-management systems.
- Validate external input.
- Use parameterized SQL queries.
- Apply least-privilege database access.
- Avoid exposing sensitive information in logs and exceptions.
- Do not disable security checks to simplify development.
- Treat Redis messages and stored objects as untrusted input.

11. Dependencies
- Keep dependencies explicit and documented.
- Avoid adding packages for functionality already available in existing dependencies.
- Avoid unnecessary dependency upgrades.
- Follow the project's dependency versioning strategy.
- Avoid importing heavyweight ML frameworks at module level unless required.
- Use lazy initialization for expensive resources.
- Keep optional dependencies optional where practical.
- Preserve CPU-only development and testing.

12. Testing and Quality Assurance
- Follow Arrange-Act-Assert.
- Test behavior rather than implementation details.
- Keep unit tests deterministic and independent.
- Test normal behavior, edge cases, and failure paths.
- Use integration tests for infrastructure interactions.
- Mock external dependencies at appropriate boundaries.
- Avoid excessive mocking of internal implementation details.
- Test idempotency and retry behavior where relevant.
- Use descriptive test names.
- Do not duplicate production logic inside tests.

13. Refactoring
- Refactor when it improves clarity, maintainability, or reliability.
- Avoid unrelated refactoring during focused bug fixes.
- Preserve existing behavior unless a change is explicitly required.
- Remove dead code when appropriate.
- Avoid introducing new architectural layers without a concrete need.
- Update tests and documentation when changing interfaces.
- Prefer incremental refactoring over large rewrites.

14. Critical Component Requirements
Apply stricter review to code involving:
- Redis message acknowledgment.
- Retry and dead-letter handling.
- Database transactions.
- Silver-to-gold transformations.
- Data persistence and consistency.
- Model output validation.
- Concurrent processing.
- Worker startup and shutdown.
- Recovery after partial failures.

For these components:
1.  Document important invariants.
2.  Make state transitions explicit.
3.  Define behavior for every relevant failure path.
4.  Avoid unbounded operations.
5.  Validate inputs and outputs.
6.  Make operations idempotent where possible.
7.  Add tests for failure and recovery scenarios.
8.  Review concurrency and data integrity risks.

15. Code Review Checklist
Before completing an implementation, verify:
- [ ] Does each component have one clear responsibility?
- [ ] Are dependencies explicit and correctly directed?
- [ ] Is business logic separated from infrastructure?
- [ ] Is the control flow easy to understand?
- [ ] Are operations bounded?
- [ ] Are inputs and outputs validated?
- [ ] Are errors handled at the correct layer?
- [ ] Are resources released correctly?
- [ ] Is processing safe under retries and concurrent execution?
- [ ] Are database and Redis contracts preserved?
- [ ] Are sensitive values protected?
- [ ] Are tests sufficient and deterministic?
- [ ] Is the implementation simpler than the alternatives?
- [ ] Did the change introduce unnecessary dependencies or abstractions?
If a solution introduces unnecessary complexity, simplify it before completing the task.

16. Exceptions to These Standards
- Treat these standards as engineering requirements.
- If a rule cannot be followed, explain why.
- Document justified architectural exceptions in an ADR when appropriate.
- Do not introduce complexity solely to satisfy a rule mechanically.
- Prefer measurable verification over subjective claims of safety.