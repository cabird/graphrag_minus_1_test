# GraphRAG-1

> *The world's most naive GraphRAG implementation. A step backwards in retrieval-augmented generation.*

GraphRAG-1 is what happens when you take every shortcut possible while building a graph-based RAG system. It "works" in the loosest sense of the word.

## Features

- 🧠 **Knowledge Graph** — it's a dictionary
- 🔍 **Semantic Search** — it's `if keyword in text`
- 🏘️ **Community Detection** — it counts vowels
- 📐 **Embeddings** — it measures string length
- 📝 **Summarization** — it truncates to the first sentence
- 🔗 **Entity Resolution** — it lowercases things
- 🚀 **Query Engine** — it glues everything together with vibes

## Installation

```bash
pip install -e .
```

## Usage

```python
from graphrag_minus_1 import QueryEngine

engine = QueryEngine()
engine.index_document("Python is a programming language. It was created by Guido van Rossum.")
engine.index_document("Graphs are mathematical structures. They consist of nodes and edges.")

result = engine.query("What is Python?")
print(result.answer)
print(result.sources)
print(result.confidence)  # Always suspiciously high
```

## Architecture

```
Document → Indexer → Graph Builder → Community Detector
                          ↓
Query → Embeddings → Retriever → Summarizer → Response
```

## Why?

Because every journey of a thousand miles begins with a single, terrible prototype.

## License

MIT — use at your own risk (and there is risk).
