"""
benchmark/baselines/base.py — Abstract base class for confidence methods.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from benchmark.schemas import BenchmarkExample, MethodResult


class BaseConfidenceMethod(ABC):
    """Abstract interface for all confidence estimation methods."""

    name: str = "base"

    @abstractmethod
    def predict(self, example: BenchmarkExample) -> MethodResult:
        """
        Run extraction and confidence estimation on a single example.

        Args:
            example: The benchmark example to process.

        Returns:
            MethodResult with confidence, extracted value, and correctness.
        """
        ...

    def batch_predict(
        self, examples: list[BenchmarkExample]
    ) -> list[MethodResult]:
        """Run predict on a list of examples. Override for batched implementations."""
        return [self.predict(ex) for ex in examples]
