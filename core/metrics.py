"""Lightweight metrics registry with a Prometheus text exporter.

Kept dependency-free on purpose: the same registry is used in-process for
tests/evaluation and can be scraped by Prometheus in production.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    # -- counters ---------------------------------------------------------- #
    def incr(self, name: str, value: float = 1.0, *, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._counters[self._label(name, labels)] += value

    def counter(self, name: str, *, labels: dict[str, str] | None = None) -> float:
        with self._lock:
            return self._counters[self._label(name, labels)]

    # -- gauges ------------------------------------------------------------ #
    def set_gauge(self, name: str, value: float, *, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._gauges[self._label(name, labels)] = value

    def gauge(self, name: str, *, labels: dict[str, str] | None = None) -> float:
        with self._lock:
            return self._gauges[self._label(name, labels)]

    # -- histograms -------------------------------------------------------- #
    def observe(self, name: str, value: float, *, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._histograms[self._label(name, labels)].append(value)

    def histogram_values(self, name: str, *, labels: dict[str, str] | None = None) -> list[float]:
        with self._lock:
            return list(self._histograms[self._label(name, labels)])

    # -- export ------------------------------------------------------------ #
    def to_prometheus_text(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                lines.append(f"# TYPE {_metric_name(name)} counter")
                lines.append(f"{name} {value}")
            for name, value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {_metric_name(name)} gauge")
                lines.append(f"{name} {value}")
            for name, values in sorted(self._histograms.items()):
                lines.append(f"# TYPE {_metric_name(name)} histogram")
                for v in values:
                    lines.append(f"{name}_bucket {v}")
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _label(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        suffix = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{suffix}}}"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: list(v) for k, v in self._histograms.items()},
            }


def _metric_name(full: str) -> str:
    return full.split("{")[0]


_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    return _registry
