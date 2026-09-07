# Pro*C Coding Rules (template)

Copy this file to `~/.config/supreme-mcp-tools/databasemcp/proc_rules.md`
and replace it with your house rules.

- Always check `sqlca.sqlcode` after every EXEC SQL statement.
- Use host variables sized to the column definitions; truncate explicitly.
- Prefer cursor-based fetch loops with a fetch limit over unbounded fetches.
- Commit per logical unit of work; never inside the fetch loop.
