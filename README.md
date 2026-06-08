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

Clone into your Claude Code plugins directory:

```bash
git clone https://github.com/gaolyLeo/repo-rag ~/.claude/plugins/repo-rag
```

Install dependencies:

```bash
pip install -r ~/.claude/plugins/repo-rag/requirements.txt
```

Claude Code will pick it up automatically via `.claude-plugin/plugin.json`.

## First run

On startup, the plugin indexes your current project in the background using [jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code). The index is saved to `.repo-rag/index.db` and reused on subsequent starts.

**Supported languages:** Python, C, C++

**Requires:** GPU recommended (RTX 4060 indexes 30k lines in ~1.4s); CPU works but is slower.

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

## Privacy

All processing is local. No code leaves your machine.
