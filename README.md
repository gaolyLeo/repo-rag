# Repo RAG

**Code-to-Code semantic search for finding similar implementations, duplicate code, and cross-file patterns.**

Unlike natural language code search tools (e.g., "find authentication code"), this focuses on **code-to-code retrieval**: given a code snippet, find semantically similar code across your repository.

## Key Features

- 🔍 **Code-to-Code Search**: Find similar implementations by providing code, not descriptions
- 🎯 **High Accuracy**: 100% for similar code, 75% for cross-file matching (validated on real repos)
- ⚡ **Blazing Fast**: 22x faster than baseline (30k lines in 1.4s on RTX 4060)
- 🔒 **Fully Local**: No data leaves your computer
- 🔌 **MCP Server**: Integrates with Claude Code

## Project Structure

```
repo-rag/
├── src/
│   └── repo_rag/          # Main source code (to be implemented)
│       ├── __init__.py
│       ├── embedding.py   # Embedding generation
│       ├── indexing.py    # Vector indexing
│       └── search.py      # Semantic search
├── benchmarks/            # Evaluation and benchmarks
│   ├── eval/              # Evaluation scripts
│   ├── tests/             # Performance tests
│   ├── data/              # Test data and chunks
│   ├── results/           # Benchmark results
│   ├── eval_report.md     # Detailed evaluation report
│   └── SUMMARY.md         # Performance summary
└── README.md
```

## Performance

**Optimized Configuration:**
- Model: `jinaai/jina-embeddings-v2-base-code`
- Precision: FP16
- Batch size: 512
- Attention: Flash Attention 2

**Speed (RTX 4060 8GB):**
- 2185 chunks/second (22x faster than baseline)
- 30k lines: 1.4 seconds
- 100k lines: 46 seconds

**Accuracy:**
- Similar code detection: 100%
- Context retrieval: 75%
- Cross-file matching: 75%

## Development

```bash
# Setup (uses shared ~/.venv)
source ~/.venv/bin/activate

# Run benchmarks
cd benchmarks
python eval/runner.py --help
```

## Benchmarks

All evaluation code and results are in the `benchmarks/` directory:
- `benchmarks/eval/` - Evaluation scripts
- `benchmarks/tests/` - Performance tests
- `benchmarks/results/` - Benchmark results
- `benchmarks/eval_report.md` - Detailed analysis

See `benchmarks/SUMMARY.md` for performance summary.
