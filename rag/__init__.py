"""A small, dependency-light RAG pipeline: embed -> store -> retrieve -> generate."""

from .pipeline import Assistant, Answer

__all__ = ["Assistant", "Answer"]
