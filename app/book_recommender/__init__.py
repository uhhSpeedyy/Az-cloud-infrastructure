"""Explainable, metadata-based book recommendations."""

from .open_library import OpenLibraryClient, OpenLibraryUnavailable
from .service import BookRecommendationService

__all__ = [
    "BookRecommendationService",
    "OpenLibraryClient",
    "OpenLibraryUnavailable",
]
