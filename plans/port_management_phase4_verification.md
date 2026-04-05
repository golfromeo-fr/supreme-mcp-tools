# Phase 4: Verification

**Goal**: Ensure no hardcoded ports remain and config loading works correctly.

---

## Task 4.1: Grep for Hardcoded Ports

Search for any remaining hardcoded port patterns:

```bash
# Search for 4-digit port literals in Python files
grep -rn "port.*=.*[0-9]\{4\}" --include="*.py" launcher/ monitoring/ mcp_ui/

# Search for old port numbers
grep -rn "9090\|9091\|9092" --include="*.py" launcher/ monitoring/ mcp_ui/
```

**Expected result**: No matches (all should be in config)

---

## Task 4.2: Test Missing Config Behavior

Test that the system fails gracefully when `ports.json` is missing:

```bash
# Temporarily rename ports.json
mv config/ports.json config/ports.json.bak

# Try to run - should fail with clear error
python launchmcp.py

# Restore
mv config/ports.json.bak config/ports.json
```

**Expected result**: Clear error message indicating `ports.json` is missing

---

## Task 4.3: Run Existing Tests

Run the test suite to ensure nothing is broken:

```bash
pytest tests/ -v
```

**Fix any failures** related to port changes

---

## Task 4.4: Manual Verification

Start the system and verify ports are allocated from config:

```bash
python launchmcp.py
# Check that:
# - Central management is on port 8200
# - Metrics is on port 8300
# - Management UI is on port 8400
# - MCP tools are on ports 8000-8004
```

---

## Verification Checklist

- [ ] No `port = <number>` patterns in Python files (except 0/None)
- [ ] No `9090`, `9091`, `9092` literals in Python files
- [ ] Missing `ports.json` produces clear error
- [ ] All services start with correct ports from config
- [ ] All existing tests pass

---

## Rollback Plan

If issues arise, rollback steps:

1. Restore `config.json` (root) from git
2. Restore `config/port_config.json` from git
3. Restore `config/launcher_config.json` from git
4. Revert Python changes in `launcher_config.py`, `port_manager.py`, etc.
5. Run tests to confirm rollback successful

---

## Dependencies

- Phase 3 should be complete before Phase 4
