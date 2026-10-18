# GuideBot — Graphify Knowledge Graph Rules

This project maintains an AST-derived knowledge graph at `graphify-out/` (179 nodes, 306 edges, 9 communities) built by [`graphify`](https://graphify.com). Use it as the primary source of codebase context before any raw source browsing.

## Mandatory graphify workflow

### 1. Query the graph before searching code

For any codebase question (where is X defined, how does A relate to B, what calls C), run:

```
graphify query "<question>"
```

**before** your first `grep`, `rg`, `read_file`, or `search_codebase`. Read raw files **only** for lines the graph or an earlier read already located.

Use:
- `graphify path "<A>" "<B>"` — to trace relationships between two symbols/files.
- `graphify explain "<concept>"` — for a focused explanation of a node and its neighbors.
- `graphify update .` — after modifying code, to rebuild the graph (AST-only, no API cost).

### 2. Prefer graph artifacts over raw browsing

- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when `query`/`path`/`explain` do not surface enough context.
- `graphify-out/graph.json` is rewritten whole on every build — a large diff is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.

### 3. Keep the graph current

After any code modification:

```
graphify update .
```

This re-extracts code files and updates the graph with no LLM/API cost. Run it before committing so the committed `graph.json` stays fresh.

### 4. Apply this rule to all subagents

Put this rule in every subagent prompt that explores code. Graphify returns a scoped subgraph, usually much smaller than `GRAPH_REPORT.md` or raw grep output, which keeps context precise and token-efficient.

## Example interactions

- "Where is `Stream` defined and what calls it?" → `graphify explain "Stream"`
- "How does `main()` connect to `InferencePipeline`?" → `graphify path "main()" "InferencePipeline"`
- "What handles Wi-Fi scanning?" → `graphify query "Wi-Fi scanning"`
- After editing `pipeline/stream.py` → `graphify update .`
