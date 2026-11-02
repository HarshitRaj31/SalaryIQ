"""
Re-export of HistoryService for app.services namespace discovery.
"""

from app.history_service import HistoryService, history_service

__all__ = ["HistoryService", "history_service"]

