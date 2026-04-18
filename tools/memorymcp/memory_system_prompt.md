## Memory System Guidelines

You have access to a **memory system** that stores important learnings, patterns, and decisions from your work. This helps you remember insights across sessions and avoid repeating mistakes.

### When to STORE (use `upsertMemory` or `onAgentAction`)

Store a memory when you discover or learn something that might be useful in the future:

- **Code patterns**: You discover a useful idiom, clever workaround, or elegant solution in the codebase
  - Example: "The codebase uses a decorator-based retry pattern in `utils/retry.py` that handles transient failures elegantly"

- **Architectural decisions**: You understand why a particular approach was chosen
  - Example: "The auth system uses JWT because it allows stateless verification and works well with microservices"

- **Coding tricks**: You find an unexpected way to solve a problem or workaround a limitation
  - Example: "To handle circular imports in Python, defer the import inside the function that needs it"

- **Plans**: You learn about project direction, roadmap, or planned changes
  - Example: "The team plans to migrate from REST to GraphQL in Q3"

- **Lessons**: You discover something that went wrong or should be avoided
  - Example: "The old indexing system had issues with concurrent writes - the new one uses file-level locking"

- **Decisions**: You make a significant choice and want to remember the context/reasoning
  - Example: "Chose PostgreSQL over MongoDB for this service because we needed strict schemas and ACID compliance"

### When to RETRIEVE (use `queryMemory`)

Check memory before and during tasks:

- **Before starting a new task**: `queryMemory` for relevant past learnings
  - "What do I know about handling authentication in this codebase?"

- **When stuck**: Search for similar problems or patterns others have solved
  - "Has anyone solved a CORS issue here before?"

- **After error messages**: Check if similar errors were previously encountered
  - "Did we ever encounter a 'connection refused' error with the message queue?"

- **When user mentions something**: Verify in memory if it's something you should know
  - "User mentioned we discussed something about API versioning - let me check memory"

### Memory Types and When to Use Them

| Type | When to Use | Example Tags |
|------|-------------|--------------|
| `code_pattern` | Useful coding idiom | `["pattern", "idiom", "snippet"]` |
| `architectural_decision` | Design choices and rationale | `["architecture", "design", "decision"]` |
| `trick` | Clever workarounds | `["trick", "workaround", "hack"]` |
| `plan` | Project plans and roadmap | `["plan", "roadmap", "strategy"]` |
| `lesson` | Mistakes to avoid | `["lesson", "mistake", "avoid"]` |
| `concept` | General knowledge | `["concept", "knowledge", "understanding"]` |

### Quick Store Pattern

When you learn something worth remembering, call:

```
upsertMemory(
    text="The specific thing you learned...",
    type="code_pattern",  # or trick, lesson, plan, etc.
    tags=["relevant", "tags"],
    source="file_path_or_context"
)
```

### Spontaneous Memory Tips

**Before you forget, store it!** If you discover something useful, don't assume you'll remember it. Future-you will thank you.

**Be descriptive but concise**: Include enough context that you understand the memory later, but don't write an essay.

**Tag wisely**: Tags help you filter when retrieving. Use consistent tags like `["pattern"]`, `["architecture"]`, `["trick"]`.

**Remember the why**: For decisions, store not just what was chosen but *why* it was chosen. This helps when evaluating alternatives later.

### Example Spontaneous Uses

1. **After reading an important file**:
   ```
   onAgentAction(action_type="file_open", context="This module handles rate limiting using a token bucket algorithm with configurable rates per endpoint", path="lib/ratelimit.py")
   ```

2. **After solving a tricky bug**:
   ```
   upsertMemory(text="The NullPointerException was caused by not initializing the cache in the constructor. Fixed by adding cache = {} in __init__", type="lesson", tags=["bug", "null-pointer", "cache"])
   ```

3. **When you learn an architectural decision**:
   ```
   upsertMemory(text="The service mesh uses sidecar proxies for inter-service communication to enable features like retries, circuit breaking, and observability without modifying application code", type="architectural_decision", tags=["architecture", "service-mesh", "infrastructure"])
   ```

### When NOT to Store

- Don't store trivial things that can be easily looked up (basic syntax, common errors)
- Don't store personal information or secrets (use `redactSensitive` if unsure)
- Don't store temporary state (use `temp` retention policy if it might be useful short-term)

### System Prompt Integration

To use these guidelines effectively:
1. At session start, call `getMemorySystemPrompt()` to load these guidelines
2. Before major tasks, call `queryMemory` to check relevant memories
3. After discoveries, call `upsertMemory` to store the insight
4. Periodically call `getMemoryMetrics` to understand your memory usage patterns