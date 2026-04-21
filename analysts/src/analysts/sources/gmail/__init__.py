"""Gmail source ingestion package."""

from .models import GmailCandidateDocument, GmailMessageRecord, GmailSyncState
from .storage import GmailStore

__all__ = [
    "GmailCandidateDocument",
    "GmailMessageRecord",
    "GmailSyncState",
    "GmailStore",
]
