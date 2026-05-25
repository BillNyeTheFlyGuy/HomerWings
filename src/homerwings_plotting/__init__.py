"""Standalone plotting and CSV analysis tools for HomerWings outputs.

This package is intentionally separate from the top-level ``HomerWings.py``
image-analysis/GUI script. It consumes the CSV files produced after analysis
and writes summary tables and plots.
"""

__all__ = ["HomerwingsDataAnalyzer"]


def __getattr__(name: str):
    if name == "HomerwingsDataAnalyzer":
        from .analyzer import HomerwingsDataAnalyzer

        return HomerwingsDataAnalyzer
    raise AttributeError(name)
