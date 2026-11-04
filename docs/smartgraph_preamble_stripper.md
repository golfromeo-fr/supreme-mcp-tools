# textToSmartGraph: Preamble Stripper — Design & Test Results

## Problem

LLMs (MiniMax M2.7, DeepSeek, OpenAI) often emit **thinking blocks** and **preamble text** before the actual structured output requested by `textToSmartGraph`. Example from MiniMax:

```
<? Let me think about this... ?>
The user wants me to compress a large knowledge graph...
Let me analyze the structure:
- This is about SymphonyAI Retail Pro*C coding rules
...
I need to:
1. Cluster related rules together
2. Compress duplicate/verbose content
...

CLUSTERS:
1. META - pctmeta1, pcmeta2 (activation/application rules)
2. CODE_STRUCTURE - pcgene1, pcgene2, commontech0...
...

Now I can compress each cluster...

CLUSTERS:
CLUSTER_NAME: META_ACTIVATION, pcgene2 — Meta rules...
```

Lines 1-46 are preamble. The real content starts at line 47.

## Solution: `strip_llm_artifacts()`

A model-agnostic post-processing function in `memorymcp_fastmcp.py` (line ~2027) that:

### Step 1: Strip thinking blocks

| Pattern | LLM | Regex |
|---|---|---|
| `<?...?>` | MiniMax | `r'<\?[\s\S]*?\?>'` |
| `<think...>...</think...>` | DeepSeek/OpenAI | `r'<think[^>]*>[\s\S]*?</think\s*>'` |
| Malformed `</think` (no `>`) | Any | `r'<think[^>]*>[\s\S]*?</think\b'` |
| Unclosed `<?...` | Truncated output | `r'<\?[\s\S]*$'` |
| Unclosed `<think...>...` | Truncated output | `r'<think[^>]*>[\s\S]*$'` |
| Incomplete `<think` (no `>`) | Truncated mid-tag | `r'<think\b[\s\S]*$'` |

### Step 2: Strip preamble before content markers

Content markers: `CLUSTERS:`, `COMPRESSED_RULES:`, `CODE:`, `##`, `**`

Algorithm:
1. Find all lines matching a content marker
2. Check each (first→last) for real content vs planning:
   - If `CLUSTERS:` followed by numbered list (`1. 2. 3.`) → planning, skip
   - If next line contains preamble phrases ("let me", "I need to", etc.) → skip
   - First marker with structured content → return from there
3. Fallback: return from last marker

## Test Results (12/12 pass)

```
PASS: MiniMax <?...?> blocks stripped
PASS: DeepSeek <think...> blocks stripped
PASS: Unclosed <think...> (truncated)
PASS: Preamble CLUSTERS: numbered list skipped
PASS: Clean output unchanged
PASS: Mixed thinking + preamble
PASS: No markers — thinking only stripped
PASS: Only thinking → empty
PASS: Empty/whitespace input
PASS: ## heading marker
PASS: ** bold marker
PASS: Actual smartgraph output — preamble stripped
```

## Files Changed

- `tools/memorymcp/memorymcp_fastmcp.py`
  - Line 19: Added `import re` at module level
  - Line ~2027: Added `_CONTENT_MARKERS`, `_PREAMBLE_PHRASES`, `strip_llm_artifacts()` function (~45 lines)
  - Line ~2182: Replaced 3-line inline `re.sub` + `import re` with `content = strip_llm_artifacts(content)`
  - Removed stale `import re` at old line 1642

- `/tmp/strip_thinking.py` — Standalone test script with 12 unit tests
