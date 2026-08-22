"""Small, defensive client for Open Library's public Search API."""

from __future__ import annotations

import copy
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import requests


OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"

# Asking for an explicit projection keeps responses small and documents every
# Open Library attribute the recommender is prepared to consume.
SEARCH_FIELDS = (
    "key",
    "title",
    "author_name",
    "author_key",
    "first_publish_year",
    "number_of_pages_median",
    "subject",
    "language",
    "edition_count",
    "ratings_average",
    "ratings_count",
    "readinglog_count",
    "want_to_read_count",
    "currently_reading_count",
    "already_read_count",
    "cover_i",
    "first_sentence",
    "editions",
    "editions.key",
    "editions.title",
    "editions.language",
)

LANGUAGE_PARAMS = {
    "ara": "ar",
    "cat": "ca",
    "ces": "cs",
    "dan": "da",
    "deu": "de",
    "ell": "el",
    "eng": "en",
    "fin": "fi",
    "fra": "fr",
    "heb": "he",
    "hin": "hi",
    "ind": "id",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "nld": "nl",
    "nor": "no",
    "pol": "pl",
    "por": "pt",
    "rus": "ru",
    "spa": "es",
    "swe": "sv",
    "tha": "th",
    "tur": "tr",
    "ukr": "uk",
    "urd": "ur",
    "vie": "vi",
    "zho": "zh",
}


def _language_param(value: str | None) -> str | None:
    normalized = str(value or "").strip().casefold().replace("_", "-")
    if not normalized:
        return None
    base = normalized.split("-", 1)[0]
    return LANGUAGE_PARAMS.get(base, base if len(base) in (2, 3) else None)


def _edition_documents(document: dict[str, Any]) -> list[dict[str, Any]]:
    editions = document.get("editions")
    if isinstance(editions, dict):
        editions = editions.get("docs")
    if not isinstance(editions, list):
        return []
    return [edition for edition in editions if isinstance(edition, dict)]


def _language_ranked_document(
    document: dict[str, Any],
    language: str | None,
) -> dict[str, Any]:
    """Use the first edition title matching the requested display language."""

    language_param = _language_param(language)
    if language_param is None:
        return document
    aliases = {language_param}
    aliases.update(
        code for code, parameter in LANGUAGE_PARAMS.items() if parameter == language_param
    )
    for edition in _edition_documents(document):
        raw_languages = edition.get("language")
        if isinstance(raw_languages, str):
            edition_languages = [raw_languages]
        elif isinstance(raw_languages, list):
            edition_languages = raw_languages
        else:
            edition_languages = []
        normalized_languages = {
            str(value).strip().casefold().removeprefix("/languages/")
            for value in edition_languages
            if isinstance(value, str)
        }
        title = edition.get("title")
        if aliases & normalized_languages and isinstance(title, str) and title.strip():
            localized = dict(document)
            localized["title"] = title
            return localized
    return document


class OpenLibraryUnavailable(RuntimeError):
    """Raised after transient Open Library failures exhaust their retries."""


@dataclass
class _CacheEntry:
    expires_at: float
    documents: list[dict[str, Any]]


class OpenLibraryClient:
    """Query Open Library with identification, retries and a TTL memory cache.

    A client instance is safe to share between Flask request threads. Successful
    responses are cached by normalized query and result limit. Callers receive a
    defensive copy so downstream feature preparation cannot mutate the cache.
    """

    def __init__(
        self,
        *,
        contact: str | None = None,
        timeout: tuple[float, float] = (2.5, 5.0),
        retries: int = 1,
        cache_ttl_seconds: float = 6 * 60 * 60,
        cache_max_entries: int = 256,
        min_request_interval: float | None = None,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative")
        if cache_max_entries < 1:
            raise ValueError("cache_max_entries must be positive")
        resolved_contact = (
            contact
            or os.environ.get("OPEN_LIBRARY_CONTACT")
            or "https://sam-speed.azurewebsites.net/"
        )
        if min_request_interval is None:
            # Open Library currently allows three identified requests/second
            # when a contact email is supplied, or one/second otherwise.
            min_request_interval = 0.36 if "@" in resolved_contact else 1.05
        if min_request_interval < 0:
            raise ValueError("min_request_interval cannot be negative")
        self.timeout = timeout
        self.retries = retries
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_max_entries = cache_max_entries
        self.min_request_interval = float(min_request_interval)
        self.session = session or requests.Session()
        self.sleeper = sleeper
        self.clock = clock
        self.user_agent = (
            "Sam-Speed-Book-Recommender/1.0 "
            f"(+{resolved_contact}; metadata search via Open Library)"
        )
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            }
        )
        self._cache: OrderedDict[tuple[str, int, str], _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._flight_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._last_request_at: float | None = None

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        language: str | None = "eng",
    ) -> list[dict[str, Any]]:
        """Return raw Search API documents for ``query``.

        ``limit`` is capped to prevent a public web request from producing an
        unexpectedly large upstream response. Network and invalid-payload errors
        are reported uniformly as :class:`OpenLibraryUnavailable`.
        """

        normalized_query = " ".join(str(query).split())
        if not normalized_query:
            return []
        bounded_limit = max(1, min(int(limit), 100))
        language_param = _language_param(language) or ""
        cache_key = (normalized_query.casefold(), bounded_limit, language_param)
        cached_documents = self._cached_documents(cache_key)
        if cached_documents is not None:
            return cached_documents

        # At most one request thread waits on Open Library. Concurrent cache
        # misses fall back locally instead of occupying every Gunicorn thread.
        if not self._flight_lock.acquire(blocking=False):
            raise OpenLibraryUnavailable("Open Library search is already in progress")
        try:
            cached_documents = self._cached_documents(cache_key)
            if cached_documents is not None:
                return cached_documents
            return self._fetch_documents(
                normalized_query,
                bounded_limit,
                cache_key,
                language_param,
            )
        finally:
            self._flight_lock.release()

    def _cached_documents(
        self,
        cache_key: tuple[str, int, str],
    ) -> list[dict[str, Any]] | None:
        now = self.clock()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and cached.expires_at > now:
                self._cache.move_to_end(cache_key)
                return copy.deepcopy(cached.documents)
            if cached:
                self._cache.pop(cache_key, None)
        return None

    def _fetch_documents(
        self,
        normalized_query: str,
        bounded_limit: int,
        cache_key: tuple[str, int, str],
        language_param: str,
    ) -> list[dict[str, Any]]:
        params = {
            "q": normalized_query,
            "fields": ",".join(SEARCH_FIELDS),
            "limit": bounded_limit,
        }
        if language_param:
            params["lang"] = language_param
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self._wait_for_rate_limit()
                response = self.session.get(
                    OPEN_LIBRARY_SEARCH_URL,
                    params=params,
                    timeout=self.timeout,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"Open Library returned HTTP {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Open Library response was not a JSON object")
                documents = payload.get("docs")
                if not isinstance(documents, list):
                    raise ValueError("Open Library response did not contain a docs list")
                clean_documents = [
                    _language_ranked_document(item, language_param)
                    for item in documents
                    if isinstance(item, dict)
                ]
                with self._lock:
                    self._cache[cache_key] = _CacheEntry(
                        expires_at=self.clock() + self.cache_ttl_seconds,
                        documents=copy.deepcopy(clean_documents),
                    )
                    self._cache.move_to_end(cache_key)
                    while len(self._cache) > self.cache_max_entries:
                        self._cache.popitem(last=False)
                return clean_documents
            except (requests.RequestException, ValueError, TypeError) as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    break
                if attempt >= self.retries:
                    break
                retry_after = self._retry_after_seconds(exc)
                if retry_after is not None and retry_after > 3.0:
                    break
                self.sleeper(
                    retry_after if retry_after is not None else 0.25 * (2**attempt)
                )

        raise OpenLibraryUnavailable("Open Library search is temporarily unavailable") from last_error

    def _wait_for_rate_limit(self) -> None:
        if self.min_request_interval <= 0:
            return
        with self._rate_lock:
            now = self.clock()
            if self._last_request_at is not None:
                delay = self.min_request_interval - (now - self._last_request_at)
                if delay > 0:
                    self.sleeper(delay)
                    refreshed = self.clock()
                    now = max(refreshed, self._last_request_at + delay)
            self._last_request_at = now

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        response = getattr(error, "response", None)
        if response is None:
            return None
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None
