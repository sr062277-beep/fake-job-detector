"""Fake Job Posting Detector: NLP + rule-based red flags + logistic regression."""
from .features import red_flags, clean_text
from .model import JobDetector

__all__ = ["JobDetector", "red_flags", "clean_text"]
__version__ = "1.0.0"
