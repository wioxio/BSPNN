"""
Custom callbacks for model training.
"""

from .early_stopping import EarlyStoppingAtMinLoss

__all__ = ["EarlyStoppingAtMinLoss"]
