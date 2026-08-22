"""Explainable content-based recommendation service.

The model deliberately uses metadata that is available before a reader rates a
book. It does not claim to analyse copyrighted full text. Style labels are
proxies inferred from subjects, descriptions and format signals, and are named
as such in every API response.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .catalogue import FALLBACK_CATALOGUE
from .open_library import OpenLibraryClient, OpenLibraryUnavailable


class _SearchClient(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        language: str | None = "eng",
    ) -> list[dict[str, Any]]: ...


THEME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Fantasy", ("fantasy", "magic", "magical realism", "dragons", "witches")),
    ("Science fiction", ("science fiction", "sci fi", "space exploration", "first contact", "aliens", "cyberpunk")),
    ("Dystopia & society", ("dystopian", "totalitarianism", "social commentary", "utopia")),
    ("Mystery & crime", ("mystery", "crime", "detective", "murder", "noir")),
    ("Thriller & suspense", ("thriller", "suspense", "spy fiction", "espionage")),
    ("Romance & relationships", ("romance", "love stories", "relationships", "marriage")),
    ("Historical fiction", ("historical fiction", "historical novel", "period fiction")),
    ("Literary & character", ("literary fiction", "character-driven", "domestic fiction")),
    ("Adventure & survival", ("adventure", "survival", "quest fiction", "action fiction")),
    ("War & conflict", ("war", "military fiction", "revolution", "conflict")),
    ("Identity & belonging", ("identity", "gender identity", "migration", "immigration", "belonging")),
    ("Family & friendship", ("family", "friendship", "found family", "family saga")),
    ("Coming of age", ("coming of age", "bildungsroman", "young adult fiction")),
    ("Mythology & folklore", ("mythology", "folklore", "legends", "fairy tales")),
    ("Politics & power", ("politics", "political fiction", "power", "totalitarianism")),
    ("Philosophy & ideas", ("philosophy", "ethics", "existential", "life choices")),
    ("Nature & environment", ("nature", "ecology", "environment", "climate")),
    ("Psychology & behaviour", ("psychology", "human behavior", "mental health", "decision making")),
    ("Science & medicine", ("medicine", "medical", "biology", "physics", "chemistry", "scientists")),
    ("History & biography", ("history", "biography", "memoir", "autobiography")),
    ("Business & finance", ("business", "finance", "economics", "money")),
    ("Self-development", ("self-help", "productivity", "habits", "personal development")),
    ("Horror & gothic", ("horror", "gothic", "ghost stories", "dark fiction")),
    ("Humour & satire", ("humorous fiction", "humor", "humour", "satire", "comedy")),
    ("Young adult", ("young adult fiction", "juvenile fiction", "teen fiction")),
)

STYLE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Immersive world-building", ("fantasy", "science fiction", "magic", "space exploration", "mythology")),
    ("Fast-paced and suspenseful", ("thriller", "suspense", "adventure", "survival", "crime", "detective")),
    ("Lyrical and atmospheric", ("lyrical fiction", "atmospheric fiction", "magical realism", "gothic fiction")),
    ("Character-driven", ("character-driven", "literary fiction", "family saga", "relationships", "coming of age")),
    ("Dark or unsettling", ("horror", "dark fiction", "dystopian", "gothic", "noir")),
    ("Witty or satirical", ("humorous fiction", "humor", "humour", "satire", "comedy")),
    ("Reflective and idea-led", ("philosophy", "ethics", "psychology", "social commentary", "memoir")),
    ("Epic and expansive", ("epic fiction", "family saga", "space opera", "high fantasy")),
    ("Romantic and emotional", ("romance", "love stories", "family", "friendship")),
    ("Investigative or analytical", ("mystery", "journalism", "medicine", "history", "biography")),
)

THEME_ORDER = {label: index for index, (label, _aliases) in enumerate(THEME_RULES)}
STYLE_ORDER = {label: index for index, (label, _aliases) in enumerate(STYLE_RULES)}

THEME_SEARCH_TERMS = {
    label: aliases[0]
    for label, aliases in THEME_RULES
}

LENGTH_BUCKETS = ("Short", "Medium", "Long", "Epic")
ERA_BUCKETS = ("Classic", "Post-war", "Modern", "Contemporary")

# Open Library normally returns ISO 639 codes, but older records also contain
# two-letter codes, bibliographic aliases and spelled-out language names. Keep
# one canonical code per language so an English work cannot accidentally match
# a Chinese-only candidate because their raw metadata happens to be noisy.
LANGUAGE_ALIASES = {
    "ar": "ara",
    "ara": "ara",
    "arabic": "ara",
    "bn": "ben",
    "ben": "ben",
    "bengali": "ben",
    "cat": "cat",
    "catalan": "cat",
    "ca": "cat",
    "chi": "zho",
    "chinese": "zho",
    "cmn": "zho",
    "mandarin": "zho",
    "zh": "zho",
    "zho": "zho",
    "cs": "ces",
    "cze": "ces",
    "ces": "ces",
    "czech": "ces",
    "da": "dan",
    "dan": "dan",
    "danish": "dan",
    "de": "deu",
    "deu": "deu",
    "ger": "deu",
    "german": "deu",
    "nl": "nld",
    "nld": "nld",
    "dut": "nld",
    "dutch": "nld",
    "el": "ell",
    "ell": "ell",
    "gre": "ell",
    "greek": "ell",
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "es": "spa",
    "spa": "spa",
    "spanish": "spa",
    "castilian": "spa",
    "fi": "fin",
    "fin": "fin",
    "finnish": "fin",
    "fr": "fra",
    "fra": "fra",
    "fre": "fra",
    "french": "fra",
    "he": "heb",
    "heb": "heb",
    "hebrew": "heb",
    "hi": "hin",
    "hin": "hin",
    "hindi": "hin",
    "id": "ind",
    "ind": "ind",
    "indonesian": "ind",
    "it": "ita",
    "ita": "ita",
    "italian": "ita",
    "ja": "jpn",
    "jpn": "jpn",
    "japanese": "jpn",
    "ko": "kor",
    "kor": "kor",
    "korean": "kor",
    "la": "lat",
    "lat": "lat",
    "latin": "lat",
    "no": "nor",
    "nor": "nor",
    "norwegian": "nor",
    "pl": "pol",
    "pol": "pol",
    "polish": "pol",
    "pt": "por",
    "por": "por",
    "portuguese": "por",
    "ru": "rus",
    "rus": "rus",
    "russian": "rus",
    "sv": "swe",
    "swe": "swe",
    "swedish": "swe",
    "th": "tha",
    "tha": "tha",
    "thai": "tha",
    "tr": "tur",
    "tur": "tur",
    "turkish": "tur",
    "uk": "ukr",
    "ukr": "ukr",
    "ukrainian": "ukr",
    "ur": "urd",
    "urd": "urd",
    "urdu": "urd",
    "vi": "vie",
    "vie": "vie",
    "vietnamese": "vie",
}
UNKNOWN_LANGUAGE_CODES = frozenset({"mis", "mul", "und", "unknown", "unspecified", "zxx"})


@dataclass(frozen=True)
class _Features:
    themes: frozenset[str]
    styles: frozenset[str]
    length: str
    era: str
    authors: frozenset[str]
    languages: frozenset[str]


@dataclass
class _ScoredBook:
    book: dict[str, Any]
    features: _Features
    score: float
    factors: dict[str, float]
    shared_themes: list[str]
    shared_styles: list[str]
    shared_authors: list[str]


def _text_key(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    return re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()


def _strings(value: Any, *, maximum: int = 80) -> list[str]:
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        cleaned = " ".join(item.split()).strip()[:500]
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
        if len(result) >= maximum:
            break
    return result


def _canonical_language(value: Any) -> str | None:
    key = _text_key(value)
    if not key:
        return None
    if key.startswith("languages "):
        key = key.removeprefix("languages ").strip()
    if key in UNKNOWN_LANGUAGE_CODES:
        return None
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    # Accommodate BCP-47 values such as ``en-US`` and labels such as
    # ``English (eng)`` without treating arbitrary prose as a language.
    for token in key.split():
        if token in UNKNOWN_LANGUAGE_CODES:
            continue
        if token in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[token]
    return key if len(key) <= 40 else None


def _language_codes(value: Any, *, maximum: int = 12) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in _strings(value, maximum=maximum):
        # Some imported records store several codes in a single field.
        parts = re.split(r"[,;|]+", raw)
        for part in parts:
            code = _canonical_language(part)
            if code and code not in seen:
                seen.add(code)
                result.append(code)
            if len(result) >= maximum:
                return result
    return result


def _uses_only_latin_letters(value: str) -> bool:
    found_letter = False
    for character in value:
        if not unicodedata.category(character).startswith("L"):
            continue
        found_letter = True
        if "LATIN" not in unicodedata.name(character, ""):
            return False
    return found_letter


def _english_author_names(
    authors: list[str],
    languages: list[str],
) -> list[str]:
    """Prefer readable Latin-script names for English records.

    Open Library work searches can return translated aliases as if they were
    separate authors (including with duplicate author keys). Keeping every
    Latin-script name preserves normal co-author records while removing those
    obvious display aliases. If no Latin form exists, the original names stay.
    """

    if languages and "eng" not in languages:
        return authors
    latin_names = [name for name in authors if _uses_only_latin_letters(name)]
    if not latin_names or len(latin_names) == len(authors):
        return authors
    return latin_names


def _integer(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _number(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _normalize_book(raw: Mapping[str, Any], *, source: str) -> dict[str, Any] | None:
    raw_title = raw.get("title")
    raw_key = raw.get("key")
    if not isinstance(raw_title, str) or not isinstance(raw_key, str):
        return None
    title = " ".join(raw_title.split()).strip()
    key = " ".join(raw_key.split()).strip()
    if not title or not key:
        return None
    if len(title) > 300 or len(key) > 160:
        return None
    if re.fullmatch(r"OL\d+W", key, flags=re.IGNORECASE):
        key = f"/works/{key.upper()}"
    if not (key.startswith("/works/") or key.startswith("/local/books/")):
        return None
    author_keys = _strings(raw.get("author_key"), maximum=12)
    languages = _language_codes(raw.get("language"), maximum=12)
    authors = _english_author_names(
        _strings(raw.get("author_name"), maximum=12),
        languages,
    )
    subjects = _strings(raw.get("subject"), maximum=80)
    first_sentences = _strings(raw.get("first_sentence"), maximum=4)
    cover_id = _integer(raw.get("cover_i"))
    curated_popularity = _number(raw.get("_curated_popularity"))
    book = {
        "key": key,
        "title": title,
        "author_name": authors,
        "author_key": author_keys,
        "first_publish_year": _bounded_integer(raw.get("first_publish_year"), 1000, 2200),
        "number_of_pages_median": _bounded_integer(raw.get("number_of_pages_median"), 1, 20_000),
        "subject": subjects,
        "language": languages,
        "edition_count": _bounded_integer(raw.get("edition_count"), 0, 10_000_000) or 0,
        "ratings_average": _bounded_number(raw.get("ratings_average"), 0.0, 5.0),
        "ratings_count": _bounded_integer(raw.get("ratings_count"), 0, 1_000_000_000) or 0,
        "readinglog_count": _bounded_integer(raw.get("readinglog_count"), 0, 1_000_000_000) or 0,
        "want_to_read_count": _bounded_integer(raw.get("want_to_read_count"), 0, 1_000_000_000) or 0,
        "currently_reading_count": _bounded_integer(raw.get("currently_reading_count"), 0, 1_000_000_000) or 0,
        "already_read_count": _bounded_integer(raw.get("already_read_count"), 0, 1_000_000_000) or 0,
        "cover_i": cover_id,
        "first_sentence": first_sentences,
        "_source": source,
        "_curated_popularity": curated_popularity,
    }
    return book


def _bounded_integer(value: Any, minimum: int, maximum: int) -> int | None:
    parsed = _integer(value)
    return parsed if parsed is not None and minimum <= parsed <= maximum else None


def _bounded_number(value: Any, minimum: float, maximum: float) -> float | None:
    parsed = _number(value)
    return parsed if parsed is not None and minimum <= parsed <= maximum else None


def _book_signature(book: Mapping[str, Any]) -> str:
    authors = book.get("author_name") or []
    first_author = authors[0] if authors else ""
    title_key = _text_key(book.get("title"))
    if not title_key:
        return f"key:{book.get('key', '')}"
    return f"{title_key}|{_text_key(first_author)}"


def _length_bucket(pages: int | None) -> str:
    if pages is None or pages <= 0:
        return "Unknown"
    if pages < 250:
        return "Short"
    if pages < 400:
        return "Medium"
    if pages < 600:
        return "Long"
    return "Epic"


def _era_bucket(year: int | None) -> str:
    if year is None or year <= 0:
        return "Unknown"
    if year < 1950:
        return "Classic"
    if year < 2000:
        return "Post-war"
    if year < 2015:
        return "Modern"
    return "Contemporary"


def _classification_scores(
    book: Mapping[str, Any],
    rules: Sequence[tuple[str, tuple[str, ...]]],
) -> dict[str, float]:
    scores: defaultdict[str, float] = defaultdict(float)
    for index, subject in enumerate(_strings(book.get("subject"), maximum=80)):
        normalized = _text_key(subject)
        if not normalized:
            continue
        padded = f" {normalized} "
        position_weight = 1.0 / (1.0 + 0.08 * index)
        for label, aliases in rules:
            evidence = 0.0
            for alias in aliases:
                normalized_alias = _text_key(alias)
                if normalized == normalized_alias:
                    evidence = max(evidence, 3.0)
                elif f" {normalized_alias} " in padded:
                    evidence = max(evidence, 1.2)
            scores[label] += evidence * position_weight
    return dict(scores)


def _salient_labels(
    scores: Mapping[str, float],
    order: Mapping[str, int],
    *,
    ratio: float,
) -> frozenset[str]:
    if not scores:
        return frozenset()
    strongest = max(scores.values())
    threshold = max(0.45, strongest * ratio)
    ranked = sorted(
        scores,
        key=lambda label: (-scores[label], order.get(label, len(order)), label),
    )
    selected = [label for label in ranked if scores[label] >= threshold][:8]
    return frozenset(selected or ranked[:1])


def _classify(book: Mapping[str, Any]) -> _Features:
    themes = set(
        _salient_labels(
            _classification_scores(book, THEME_RULES),
            THEME_ORDER,
            ratio=0.20,
        )
    )
    styles = set(
        _salient_labels(
            _classification_scores(book, STYLE_RULES),
            STYLE_ORDER,
            ratio=0.16,
        )
    )
    pages = _integer(book.get("number_of_pages_median"))
    if pages is not None and pages >= 650:
        styles.add("Epic and expansive")
    authors = frozenset(
        _text_key(value)
        for value in book.get("author_name", [])
        if _text_key(value)
    )
    return _Features(
        themes=frozenset(themes),
        styles=frozenset(styles),
        length=_length_bucket(pages),
        era=_era_bucket(_integer(book.get("first_publish_year"))),
        authors=authors,
        languages=frozenset(_language_codes(book.get("language"), maximum=12)),
    )


def _normalized_counter(
    counter: Counter[str],
    preferred_order: Sequence[str] = (),
) -> dict[str, float]:
    total = float(sum(value for value in counter.values() if value > 0))
    if not total:
        return {}
    order = {value: index for index, value in enumerate(preferred_order)}
    return {
        key: value / total
        for key, value in sorted(
            counter.items(),
            key=lambda item: (-item[1], order.get(item[0], len(order)), item[0]),
        )
        if value > 0
    }


def _canonical_option(value: Any, choices: Sequence[str]) -> str | None:
    normalized = _text_key(value)
    return next((choice for choice in choices if _text_key(choice) == normalized), None)


def _canonical_themes(values: Any) -> set[str]:
    requested = {_text_key(value) for value in _strings(values, maximum=12)}
    result: set[str] = set()
    for label, aliases in THEME_RULES:
        possibilities = {_text_key(label), *(_text_key(alias) for alias in aliases)}
        if requested & possibilities:
            result.add(label)
    return result


def _profile_similarity(distribution: Mapping[str, float], value: str, order: Sequence[str]) -> float:
    if value == "Unknown" or not distribution:
        return 0.35
    if value in distribution:
        exact = distribution[value]
    else:
        exact = 0.0
    if value not in order:
        return exact
    index = order.index(value)
    neighbours = {
        order[position]
        for position in (index - 1, index + 1)
        if 0 <= position < len(order)
    }
    nearby = sum(distribution.get(item, 0.0) for item in neighbours)
    return min(1.0, exact + 0.55 * nearby)


def _popularity(book: Mapping[str, Any]) -> float:
    curated = _number(book.get("_curated_popularity"))
    ratings = max(0, _integer(book.get("ratings_count")) or 0)
    reading = max(0, _integer(book.get("readinglog_count")) or 0)
    if not reading:
        reading = max(
            0,
            (_integer(book.get("want_to_read_count")) or 0)
            + (_integer(book.get("currently_reading_count")) or 0)
            + (_integer(book.get("already_read_count")) or 0),
        )
    editions = max(0, _integer(book.get("edition_count")) or 0)
    if not (ratings or reading or editions):
        return max(0.0, min(curated, 1.0)) if curated is not None else 0.30
    return min(
        1.0,
        0.55 * math.log1p(ratings) / math.log1p(2_000_000)
        + 0.30 * math.log1p(reading) / math.log1p(4_000_000)
        + 0.15 * math.log1p(editions) / math.log1p(10_000),
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


class BookRecommendationService:
    """Search for favourite books and produce explainable recommendations.

    Parameters are injectable so tests and offline deployments never need live
    network access. With the defaults, Open Library enriches both search and the
    recommendation candidate pool; any failure falls back to the local catalogue.

    Public methods:

    * ``search(query, limit=8) -> dict``
    * ``recommend(favourites, preferences=None) -> dict``
    """

    MODEL_VERSION = "metadata-content-v1"

    def __init__(
        self,
        *,
        client: _SearchClient | None = None,
        catalogue: Iterable[Mapping[str, Any]] | None = None,
        remote_recommendations: bool = True,
        remote_candidate_limit: int = 60,
    ) -> None:
        self.client = client or OpenLibraryClient()
        self.remote_recommendations = bool(remote_recommendations)
        self.remote_candidate_limit = max(10, min(int(remote_candidate_limit), 100))
        source_catalogue = FALLBACK_CATALOGUE if catalogue is None else catalogue
        self._catalogue = [
            book
            for raw in source_catalogue
            if (book := _normalize_book(raw, source="curated_fallback")) is not None
        ]

    def search(self, query: str, limit: int = 8) -> dict[str, Any]:
        """Search canonical Open Library works, supplemented by local matches.

        The response is directly serializable and every result can be passed back
        unchanged as a favourite to :meth:`recommend`.
        """

        normalized_query = " ".join(str(query or "").split())
        bounded_limit = max(1, min(int(limit), 20))
        if not normalized_query:
            return {"query": "", "results": [], "source": "none"}

        remote: list[dict[str, Any]] = []
        remote_failed = False
        try:
            remote = [
                book
                for raw in self.client.search(normalized_query, limit=min(40, bounded_limit * 2))
                if (book := _normalize_book(raw, source="open_library")) is not None
            ]
        except OpenLibraryUnavailable:
            remote_failed = True

        local_matches = self._local_search(normalized_query)
        merged = self._merge_books([*remote, *local_matches])
        results = merged[:bounded_limit]
        used_sources = {book["_source"] for book in results}
        if used_sources == {"open_library"}:
            source = "open_library"
        elif used_sources == {"curated_fallback"}:
            source = "curated_fallback"
        elif used_sources:
            source = "open_library+curated_fallback"
        else:
            source = "curated_fallback" if remote_failed else "open_library"
        return {
            "query": normalized_query,
            "results": [self._public_book(book) for book in results],
            "source": source,
            "degraded": remote_failed,
        }

    def recommend(
        self,
        favourites: Sequence[Mapping[str, Any]],
        preferences: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Recommend unseen works based on favourite-book metadata.

        ``preferences`` may contain ``include_themes``, ``exclude_themes``,
        ``preferred_length``, ``ignore_length``, ``preferred_era``, ``language``,
        ``allow_same_author``, ``author_emphasis``, ``shortlist_size`` and
        ``theme_list_count``.
        Unknown preference keys are ignored for forward compatibility.
        """

        if not isinstance(favourites, Sequence) or isinstance(favourites, (str, bytes)):
            raise ValueError("favourites must be a list of book objects")
        if not favourites:
            raise ValueError("at least one favourite book is required")
        if len(favourites) > 20:
            raise ValueError("at most 20 favourite books can be used at once")
        if preferences is not None and not isinstance(preferences, Mapping):
            raise ValueError("preferences must be an object")
        options = dict(preferences or {})

        favourite_books: list[dict[str, Any]] = []
        for index, raw in enumerate(favourites):
            if not isinstance(raw, Mapping):
                raise ValueError(f"favourite at index {index} must be a book object")
            book = _normalize_book(raw, source="favourite")
            if book is None:
                raise ValueError(f"favourite at index {index} requires key and title")
            favourite_books.append(book)
        favourite_books = self._merge_books(favourite_books)

        profile = self._build_profile(favourite_books, options)
        remote_books = self._recommendation_search(profile) if self.remote_recommendations else []
        candidates = self._merge_books([*remote_books, *self._catalogue])

        selected_keys = {book["key"] for book in favourite_books}
        selected_signatures = {_book_signature(book) for book in favourite_books}
        excluded_themes = _canonical_themes(options.get("exclude_themes"))
        allow_same_author = bool(options.get("allow_same_author", True))
        favourite_authors = set(profile["author_weights"])
        target_languages = set(profile["target_languages"])
        eligible: list[dict[str, Any]] = []
        for book in candidates:
            features = _classify(book)
            if book["key"] in selected_keys or _book_signature(book) in selected_signatures:
                continue
            if target_languages and not (target_languages & features.languages):
                continue
            if excluded_themes & features.themes:
                continue
            if not allow_same_author and favourite_authors & features.authors:
                continue
            book["_features"] = features
            eligible.append(book)

        scored = [self._score_book(book, profile) for book in eligible]
        scored.sort(key=lambda item: (-item.score, item.book["title"].casefold(), item.book["key"]))

        requested_shortlist_size = _integer(options.get("shortlist_size"))
        shortlist_size = max(2, min(6 if requested_shortlist_size is None else requested_shortlist_size, 8))
        requested_theme_count = _integer(options.get("theme_list_count"))
        theme_list_count = max(0, min(3 if requested_theme_count is None else requested_theme_count, 4))
        overall = self._diverse_selection(scored, shortlist_size)
        shortlists: list[dict[str, Any]] = [
            {
                "name": "Best overall",
                "kind": "overall",
                "basis": "Balanced fit across themes, style proxies, length, era, author and popularity.",
                "books": [self._recommendation_book(item, profile) for item in overall],
            }
        ]

        used_book_keys = {item.book["key"] for item in overall}
        theme_sources: dict[str, list[str]] = profile["theme_sources"]
        for theme in list(profile["theme_weights"])[:theme_list_count]:
            theme_candidates = [
                item
                for item in scored
                if theme in item.features.themes and item.book["key"] not in used_book_keys
            ]
            if not theme_candidates:
                continue
            chosen = self._diverse_selection(theme_candidates, min(4, shortlist_size))
            if len(chosen) < 2:
                continue
            used_book_keys.update(item.book["key"] for item in chosen)
            supporting_titles = theme_sources.get(theme, [])
            shortlists.append(
                {
                    "name": f"More {theme}",
                    "kind": "theme",
                    "theme": theme,
                    "basis": {
                        "favourite_titles": supporting_titles,
                        "favourite_count": len(supporting_titles),
                    },
                    "books": [self._recommendation_book(item, profile) for item in chosen],
                }
            )

        source = "open_library+curated_fallback" if remote_books else "curated_fallback"
        return {
            "model": {
                "name": "Explainable metadata content model",
                "version": self.MODEL_VERSION,
                "candidate_source": source,
            },
            "favourites": [self._public_book(book) for book in favourite_books],
            "taste_profile": self._public_profile(profile),
            "shortlists": shortlists,
            "notices": [
                "Style is a proxy inferred from subjects and descriptive metadata; the model does not analyse full book text.",
                "Recommendations are similarities, not guarantees that every reader will enjoy a book.",
            ],
        }

    def _local_search(self, query: str) -> list[dict[str, Any]]:
        normalized = _text_key(query)
        if not normalized:
            return []
        tokens = set(normalized.split())
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for book in self._catalogue:
            title = _text_key(book["title"])
            authors = " ".join(_text_key(value) for value in book["author_name"])
            subjects = " ".join(_text_key(value) for value in book["subject"])
            score = 0.0
            if title == normalized:
                score += 100.0
            elif title.startswith(normalized):
                score += 85.0
            elif normalized in title:
                score += 70.0
            if normalized in authors:
                score += 55.0
            title_tokens = set(title.split())
            author_tokens = set(authors.split())
            if tokens:
                score += 35.0 * len(tokens & title_tokens) / len(tokens)
                score += 20.0 * len(tokens & author_tokens) / len(tokens)
            if normalized in subjects:
                score += 20.0
            if score > 0:
                ranked.append((score, title, book))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]["key"]))
        return [item[2] for item in ranked]

    @staticmethod
    def _merge_books(books: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        signature_indexes: dict[str, int] = {}
        key_indexes: dict[str, int] = {}
        for book in books:
            signature = _book_signature(book)
            existing_index = key_indexes.get(book["key"], signature_indexes.get(signature))
            if existing_index is None:
                copy_book = dict(book)
                merged.append(copy_book)
                index = len(merged) - 1
                key_indexes[copy_book["key"]] = index
                signature_indexes[signature] = index
                continue
            existing = merged[existing_index]
            # Prefer the canonical Open Library identity and popularity fields,
            # while retaining richer curated subjects/pages when they are absent.
            primary, supplemental = (
                (book, existing)
                if book.get("_source") == "open_library"
                else (existing, book)
            )
            combined = dict(primary)
            for field in (
                "author_name",
                "author_key",
                "subject",
                "language",
                "first_sentence",
                "first_publish_year",
                "number_of_pages_median",
                "cover_i",
                "ratings_average",
                "_curated_popularity",
            ):
                if not combined.get(field) and supplemental.get(field):
                    combined[field] = supplemental[field]
            for field in (
                "edition_count",
                "ratings_count",
                "readinglog_count",
                "want_to_read_count",
                "currently_reading_count",
                "already_read_count",
            ):
                combined[field] = max(_integer(primary.get(field)) or 0, _integer(supplemental.get(field)) or 0)
            old_key = existing["key"]
            merged[existing_index] = combined
            key_indexes.pop(old_key, None)
            key_indexes[combined["key"]] = existing_index
            signature_indexes[signature] = existing_index
        return merged

    def _build_profile(
        self,
        favourites: Sequence[dict[str, Any]],
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        theme_counts: Counter[str] = Counter()
        style_counts: Counter[str] = Counter()
        length_counts: Counter[str] = Counter()
        era_counts: Counter[str] = Counter()
        author_counts: Counter[str] = Counter()
        observed_language_counts: Counter[str] = Counter()
        favourite_language_sets: list[frozenset[str]] = []
        theme_sources: defaultdict[str, list[str]] = defaultdict(list)
        known_pages: list[int] = []
        display_authors: dict[str, str] = {}

        for book in favourites:
            features = _classify(book)
            theme_share = 1.0 / len(features.themes) if features.themes else 0.0
            for theme in features.themes:
                theme_counts[theme] += theme_share
                theme_sources[theme].append(book["title"])
            style_share = 1.0 / len(features.styles) if features.styles else 0.0
            for style in features.styles:
                style_counts[style] += style_share
            if features.length != "Unknown":
                length_counts[features.length] += 1
            if features.era != "Unknown":
                era_counts[features.era] += 1
            for raw_author in book["author_name"]:
                key = _text_key(raw_author)
                if key:
                    author_counts[key] += 1
                    display_authors.setdefault(key, raw_author)
            favourite_language_sets.append(features.languages)
            observed_language_counts.update(features.languages)
            pages = _integer(book.get("number_of_pages_median"))
            if pages and pages > 0:
                known_pages.append(pages)

        included_themes = _canonical_themes(options.get("include_themes"))
        for theme in included_themes:
            theme_counts[theme] += 1.5
            theme_sources[theme].append("Your explicit preference")
        preferred_length = _canonical_option(options.get("preferred_length"), LENGTH_BUCKETS)
        if preferred_length:
            length_counts[preferred_length] += 2
        if bool(options.get("ignore_length")):
            length_counts.clear()
        preferred_eras = [
            era
            for value in _strings(options.get("preferred_era"), maximum=2)
            if (era := _canonical_option(value, ERA_BUCKETS))
        ]
        for preferred_era in preferred_eras:
            era_counts[preferred_era] += 2 / len(preferred_eras)
        requested_languages = _language_codes(options.get("language"), maximum=4)
        if requested_languages:
            target_languages = tuple(requested_languages)
        elif any(not languages for languages in favourite_language_sets) or any(
            "eng" in languages for languages in favourite_language_sets
        ):
            # Missing edition metadata defaults to English for this English UI.
            target_languages = ("eng",)
        elif observed_language_counts:
            strongest = max(observed_language_counts.values())
            target_languages = tuple(
                sorted(
                    language
                    for language, count in observed_language_counts.items()
                    if count == strongest
                )
            )
        else:
            target_languages = ("eng",)
        language_counts = Counter(
            {
                language: max(1, observed_language_counts.get(language, 0))
                for language in target_languages
            }
        )
        author_emphasis = max(1.0, min(_number(options.get("author_emphasis")) or 1.0, 2.0))

        return {
            "theme_weights": _normalized_counter(
                theme_counts,
                [label for label, _aliases in THEME_RULES],
            ),
            "style_weights": _normalized_counter(
                style_counts,
                [label for label, _aliases in STYLE_RULES],
            ),
            "length_weights": _normalized_counter(length_counts, LENGTH_BUCKETS),
            "era_weights": _normalized_counter(era_counts, ERA_BUCKETS),
            "author_weights": _normalized_counter(author_counts),
            "language_weights": _normalized_counter(language_counts),
            "target_languages": target_languages,
            "theme_sources": {
                theme: list(dict.fromkeys(titles))
                for theme, titles in sorted(theme_sources.items())
            },
            "display_authors": display_authors,
            "average_pages": round(sum(known_pages) / len(known_pages)) if known_pages else None,
            "author_emphasis": author_emphasis,
        }

    def _recommendation_search(self, profile: Mapping[str, Any]) -> list[dict[str, Any]]:
        top_themes = list(profile["theme_weights"])[:3]
        if not top_themes:
            return []
        terms = [THEME_SEARCH_TERMS[theme] for theme in top_themes]
        theme_query = " OR ".join(f'subject:\"{term}\"' for term in terms)
        target_languages = list(profile.get("target_languages", ()))
        if target_languages:
            language_query = " OR ".join(
                f"language:{language}" for language in target_languages
            )
            query = f"({theme_query}) AND ({language_query})"
        else:
            query = theme_query
        try:
            documents = self.client.search(
                query,
                limit=self.remote_candidate_limit,
                language=target_languages[0] if target_languages else "eng",
            )
        except OpenLibraryUnavailable:
            return []
        return [
            book
            for raw in documents
            if (book := _normalize_book(raw, source="open_library")) is not None
        ]

    def _score_book(self, book: dict[str, Any], profile: Mapping[str, Any]) -> _ScoredBook:
        features: _Features = book["_features"]
        theme_weights: Mapping[str, float] = profile["theme_weights"]
        style_weights: Mapping[str, float] = profile["style_weights"]
        shared_themes = sorted(
            features.themes & theme_weights.keys(),
            key=lambda item: (-theme_weights[item], THEME_ORDER.get(item, len(THEME_ORDER)), item),
        )
        shared_styles = sorted(
            features.styles & style_weights.keys(),
            key=lambda item: (-style_weights[item], STYLE_ORDER.get(item, len(STYLE_ORDER)), item),
        )
        shared_authors = sorted(features.authors & profile["author_weights"].keys())
        theme_score = min(1.0, sum(theme_weights[item] for item in shared_themes))
        style_score = min(1.0, sum(style_weights[item] for item in shared_styles))
        length_score = _profile_similarity(profile["length_weights"], features.length, LENGTH_BUCKETS)
        era_score = _profile_similarity(profile["era_weights"], features.era, ERA_BUCKETS)
        author_score = min(
            1.0,
            sum(profile["author_weights"].get(author, 0.0) for author in shared_authors),
        )
        popularity_score = _popularity(book)
        language_weights: Mapping[str, float] = profile["language_weights"]
        if not language_weights or not features.languages:
            language_score = 0.6
        else:
            language_score = min(1.0, sum(language_weights.get(item, 0.0) for item in features.languages))
        factors = {
            "themes": theme_score,
            "style_proxy": style_score,
            "length": length_score,
            "era": era_score,
            "author": author_score,
            "popularity": popularity_score,
            "language": language_score,
        }
        author_emphasis = profile.get("author_emphasis", 1.0)
        affinity = (
            0.38 * theme_score
            + 0.18 * style_score
            + 0.12 * length_score
            + 0.10 * era_score
            + 0.08 * author_score
            + 0.10 * popularity_score
            + 0.04 * language_score
        )
        affinity += 0.08 * (author_emphasis - 1.0) * author_score
        score = 100.0 * min(1.0, affinity)
        return _ScoredBook(
            book=book,
            features=features,
            score=score,
            factors=factors,
            shared_themes=shared_themes,
            shared_styles=shared_styles,
            shared_authors=shared_authors,
        )

    @staticmethod
    def _diverse_selection(scored: Sequence[_ScoredBook], limit: int) -> list[_ScoredBook]:
        remaining = list(scored)
        selected: list[_ScoredBook] = []
        while remaining and len(selected) < limit:
            selected_authors = (
                set().union(*(item.features.authors for item in selected))
                if selected
                else set()
            )
            viable = [
                item
                for item in remaining
                if not (item.features.authors & selected_authors)
            ]
            if not viable:
                break

            def adjusted(item: _ScoredBook) -> tuple[float, str, str]:
                redundancy = 0.0
                for chosen in selected:
                    overlap = _jaccard(item.features.themes, chosen.features.themes)
                    same_author = bool(item.features.authors & chosen.features.authors)
                    redundancy = max(redundancy, 8.0 * overlap + (4.0 if same_author else 0.0))
                return (
                    -(item.score - redundancy),
                    item.book["title"].casefold(),
                    item.book["key"],
                )

            chosen = min(viable, key=adjusted)
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def _recommendation_book(
        self,
        item: _ScoredBook,
        profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = self._public_book(item.book)
        result.update(
            {
                "match_score": round(item.score, 1),
                "reasons": self._reasons(item, profile),
                "matched_themes": item.shared_themes[:4],
                "style_proxies": item.shared_styles[:3],
                "length_band": item.features.length,
                "era": item.features.era,
                "match_factors": {
                    key: round(value * 100.0, 1)
                    for key, value in item.factors.items()
                },
            }
        )
        return result

    @staticmethod
    def _reasons(item: _ScoredBook, profile: Mapping[str, Any]) -> list[str]:
        reasons: list[str] = []
        if item.shared_themes:
            labels = item.shared_themes[:2]
            reasons.append(f"Shares your interest in {' and '.join(labels)}.")
        if item.shared_styles:
            reasons.append(f"Its metadata suggests a similar {item.shared_styles[0].casefold()} feel.")
        if item.shared_authors:
            display = profile["display_authors"].get(item.shared_authors[0], item.shared_authors[0])
            reasons.append(f"Another work by {display}.")
        if item.factors["length"] >= 0.75 and item.features.length != "Unknown":
            reasons.append(f"Its {item.features.length.casefold()} length matches your usual range.")
        if item.factors["era"] >= 0.75 and item.features.era != "Unknown":
            reasons.append(f"It fits your interest in {item.features.era.casefold()} books.")
        if item.factors["popularity"] >= 0.85:
            reasons.append("It is also a widely read candidate.")
        if not reasons:
            reasons.append("It is a broad metadata match with solid reader interest.")
        return reasons[:3]

    @staticmethod
    def _public_book(book: Mapping[str, Any]) -> dict[str, Any]:
        cover_id = _integer(book.get("cover_i"))
        if cover_id is not None and cover_id <= 0:
            cover_id = None
        key = str(book["key"])
        return {
            "key": key,
            "title": book["title"],
            "author_name": list(book.get("author_name", [])),
            "author_key": list(book.get("author_key", [])),
            "first_publish_year": book.get("first_publish_year"),
            "number_of_pages_median": book.get("number_of_pages_median"),
            "subject": list(book.get("subject", []))[:30],
            "language": list(book.get("language", [])),
            "first_sentence": list(book.get("first_sentence", []))[:4],
            "edition_count": book.get("edition_count", 0),
            "ratings_average": book.get("ratings_average"),
            "ratings_count": book.get("ratings_count", 0),
            "readinglog_count": book.get("readinglog_count", 0),
            "want_to_read_count": book.get("want_to_read_count", 0),
            "currently_reading_count": book.get("currently_reading_count", 0),
            "already_read_count": book.get("already_read_count", 0),
            "cover_i": cover_id,
            "cover_url": (
                f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
                if cover_id is not None
                else None
            ),
            "open_library_url": (
                f"https://openlibrary.org{key}"
                if key.startswith("/works/")
                else None
            ),
            "source": book.get("_source", "unknown"),
        }

    @staticmethod
    def _public_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "themes": [
                {
                    "name": name,
                    "weight": round(weight, 3),
                    "grounded_in": profile["theme_sources"].get(name, []),
                }
                for name, weight in list(profile["theme_weights"].items())[:8]
            ],
            "style_proxies": [
                {"name": name, "weight": round(weight, 3)}
                for name, weight in list(profile["style_weights"].items())[:6]
            ],
            "length": {
                "dominant_band": next(iter(profile["length_weights"]), "Unknown"),
                "average_pages": profile["average_pages"],
                "distribution": {
                    name: round(weight, 3)
                    for name, weight in profile["length_weights"].items()
                },
            },
            "eras": [
                {"name": name, "weight": round(weight, 3)}
                for name, weight in profile["era_weights"].items()
            ],
            "authors": [
                {
                    "name": profile["display_authors"].get(name, name),
                    "weight": round(weight, 3),
                }
                for name, weight in list(profile["author_weights"].items())[:6]
            ],
            "style_note": "Style labels are metadata-derived proxies, not full-text prose analysis.",
        }
