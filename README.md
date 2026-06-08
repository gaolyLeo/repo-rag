# repo-rag

![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-orange?logo=anthropic)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Local](https://img.shields.io/badge/Privacy-100%25%20Local-brightgreen)
![GPU](https://img.shields.io/badge/GPU-FP16%20%2B%20Flash%20Attention%202-76b900?logo=nvidia)

A **Claude Code plugin** that gives Claude semantic code search over your repository.

When installed, Claude can call `search_code` to find implementations similar to a code snippet — across your entire codebase, fully locally, without sending code anywhere.

## What it does

Claude Code generates a "what the code I'm looking for might look like" snippet and passes it to `search_code`. The plugin returns the most similar chunks with file paths and line numbers.

This is **code-to-code retrieval**, not natural language search. You don't use it directly — Claude does.

## Installation

### Prerequisites

- [Claude Code](https://claude.ai/code) installed
- Python 3.10+
- GPU strongly recommended (NVIDIA, with CUDA). CPU works but embedding is slow.

### Step 1 — Clone the plugin

```bash
git clone https://github.com/gaolyLeo/repo-rag ~/.claude/plugins/repo-rag
```

Claude Code automatically discovers plugins placed under `~/.claude/plugins/`.

### Step 2 — Install dependencies

```bash
pip install -r ~/.claude/plugins/repo-rag/requirements.txt
```

> **Note:** `flash_attn` requires CUDA (NVIDIA GPU). On **Mac (Apple Silicon)**, skip it — the plugin uses MPS automatically:
> ```bash
> pip install -r ~/.claude/plugins/repo-rag/requirements.txt --ignore-requires-python
> # or just remove the flash_attn line before installing
> ```
> On CPU-only machines, remove `flash_attn` from `requirements.txt` before installing.

### Step 3 — Verify

Open Claude Code in any project directory. You should see `repo-rag` listed in the active MCP servers. On first launch, Claude will say the index is building in the background — this can take up to a minute for large repos.

That's it. No config files to edit, no API keys.

### How Claude uses it

Once installed, Claude automatically calls `search_code` when it needs to find similar code in your project. You don't invoke it manually — just ask Claude things like:

- *"Is there already a retry mechanism somewhere in this codebase?"*
- *"Find all places that handle JWT validation"*
- *"Are there other functions that do the same thing as this one?"*

Claude will generate a representative code snippet internally and use it as the search query.

### Index location

The index is stored at `.repo-rag/index.db` inside your project. It is reused on subsequent Claude Code sessions — no rebuild needed unless you add `.repo-rag/` to `.gitignore` (recommended, it's already there if you cloned this repo as a template).

**Supported languages:** Python, C, C++

## Performance

Benchmarked on an RTX 4060 8GB:

| Metric | Result |
|--------|--------|
| Speed | 2185 chunks/s (22× faster than baseline) |
| Similar code detection | 100% |
| Cross-file matching | 75% |

## Project structure

```
src/
├── server.py          # MCP server entry point
├── builder.py         # Repo walker + index builder
├── chunking/          # Tree-sitter based code chunker
└── indexing/          # Embedder, vector index (sqlite-vec), searcher
```

## Roadmap

- [ ] **Incremental indexing** — only re-embed files changed since last build, instead of full rebuild
- [ ] **More languages** — TypeScript, Go, Rust, Java (tree-sitter adapters)
- [ ] **Watch mode** — auto-reindex on file save during active Claude Code sessions
- [ ] **Multi-repo search** — search across multiple indexed repos at once
- [ ] **Chunk deduplication** — skip near-identical chunks to reduce index bloat

## Privacy

All processing is local. No code leaves your machine.
