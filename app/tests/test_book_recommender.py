from __future__ import annotations

import json
import threading
from typing import Any

import pytest
import requests

from book_recommender import (
    BookRecommendationService,
    OpenLibraryClient,
    OpenLibraryUnavailable,
)
from book_recommender.open_library import SEARCH_FIELDS


DUNE = {
    "key": "/works/OL-TEST-DUNE-W",
    "title": "Dune",
    "author_name": ["Frank Herbert"],
    "first_publish_year": 1965,
    "number_of_pages_median": 688,
    "subject": [
        "Science fiction",
        "Adventure",
        "Politics",
        "Ecology",
        "Epic fiction",
    ],
    "language": ["eng"],
}

THE_HOBBIT = {
    "key": "/works/OL-TEST-HOBBIT-W",
    "title": "The Hobbit",
    "author_name": ["J. R. R. Tolkien"],
    "first_publish_year": 1937,
    "number_of_pages_median": 310,
    "subject": ["Fantasy", "Adventure", "Mythology", "Dragons", "Quest fiction"],
    "language": ["eng"],
}


class _UnavailableClient:
    def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        raise OpenLibraryUnavailable("offline in test")


class _NoNetworkClient:
    def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        raise AssertionError("this test must not use the network client")


class _StaticClient:
    def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return [
            {
                "key": "OL123W",
                "title": "Remote Result",
                "author_name": ["Remote Author"],
                "first_publish_year": 2021,
                "subject": ["Science fiction"],
            }
        ]


class _Response:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=self,
            )

    def json(self) -> Any:
        return self._payload


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.headers: dict[str, str] = {}
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def test_open_library_client_identifies_retries_projects_fields_and_caches():
    document = {
        "key": "/works/OL1W",
        "title": "A Test Book",
        "author_name": ["An Author"],
    }
    session = _Session(
        [
            _Response(503, {}),
            _Response(200, {"docs": [document]}),
        ]
    )
    sleeps: list[float] = []
    client = OpenLibraryClient(
        session=session,  # type: ignore[arg-type]
        retries=1,
        min_request_interval=0,
        sleeper=sleeps.append,
        clock=lambda: 100.0,
    )

    first = client.search("  test   book ", limit=8)
    first[0]["title"] = "mutated by caller"
    second = client.search("test book", limit=8)

    assert len(session.calls) == 2
    assert session.calls[0]["timeout"] == (2.5, 5.0)
    assert session.calls[0]["params"]["fields"] == ",".join(SEARCH_FIELDS)
    assert session.calls[0]["params"]["q"] == "test book"
    assert session.calls[0]["params"]["lang"] == "en"
    assert "Sam-Speed-Book-Recommender" in session.headers["User-Agent"]
    assert sleeps == [0.25]
    assert second[0]["title"] == "A Test Book"


def test_open_library_uses_matching_edition_title_for_requested_language():
    work = {
        "key": "/works/OLWITCHERW",
        "title": "Wieża jaskółki",
        "language": ["pol", "eng", "fre"],
        "editions": {
            "docs": [
                {
                    "key": "/books/OLPOLM",
                    "title": "Wieża jaskółki",
                    "language": ["pol"],
                },
                {
                    "key": "/books/OLENM",
                    "title": "The Tower of the Swallow",
                    "language": ["eng"],
                },
                {
                    "key": "/books/OLFRM",
                    "title": "La Tour de l'Hirondelle",
                    "language": ["fra"],
                },
            ]
        },
    }
    session = _Session(
        [
            _Response(200, {"docs": [work]}),
            _Response(200, {"docs": [work]}),
        ]
    )
    client = OpenLibraryClient(
        session=session,  # type: ignore[arg-type]
        retries=0,
        min_request_interval=0,
    )

    english = client.search("witcher", language="eng")
    french = client.search("witcher", language="fra")

    assert english[0]["key"] == "/works/OLWITCHERW"
    assert english[0]["title"] == "The Tower of the Swallow"
    assert french[0]["key"] == "/works/OLWITCHERW"
    assert french[0]["title"] == "La Tour de l'Hirondelle"
    assert session.calls[0]["params"]["lang"] == "en"
    assert session.calls[1]["params"]["lang"] == "fr"
    assert "editions.title" in session.calls[0]["params"]["fields"]


def test_open_library_client_observes_the_default_public_rate_limit():
    session = _Session(
        [
            _Response(200, {"docs": []}),
            _Response(200, {"docs": []}),
        ]
    )
    timeline = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        timeline[0] += seconds

    client = OpenLibraryClient(
        session=session,  # type: ignore[arg-type]
        retries=0,
        sleeper=sleep,
        clock=lambda: timeline[0],
    )

    client.search("first query")
    client.search("second query")

    assert sleeps == [1.05]


def test_open_library_client_rejects_invalid_top_level_json():
    for payload in (None, [], {"docs": "not a list"}):
        client = OpenLibraryClient(
            session=_Session([_Response(200, payload)]),  # type: ignore[arg-type]
            retries=0,
            min_request_interval=0,
        )

        with pytest.raises(OpenLibraryUnavailable):
            client.search("Dune")


def test_open_library_client_honours_long_retry_after_without_retrying_early():
    session = _Session(
        [
            _Response(429, {}, headers={"Retry-After": "120"}),
            _Response(200, {"docs": []}),
        ]
    )
    sleeps: list[float] = []
    client = OpenLibraryClient(
        session=session,  # type: ignore[arg-type]
        retries=1,
        min_request_interval=0,
        sleeper=sleeps.append,
    )

    with pytest.raises(OpenLibraryUnavailable):
        client.search("Dune")

    assert len(session.calls) == 1
    assert sleeps == []


def test_open_library_client_allows_only_one_in_flight_cache_miss():
    started = threading.Event()
    release = threading.Event()

    class _BlockingSession(_Session):
        def get(self, url: str, **kwargs: Any) -> _Response:
            self.calls.append({"url": url, **kwargs})
            started.set()
            assert release.wait(timeout=2)
            return self.responses.pop(0)

    session = _BlockingSession([_Response(200, {"docs": []})])
    client = OpenLibraryClient(
        session=session,  # type: ignore[arg-type]
        retries=0,
        min_request_interval=0,
    )
    failures: list[BaseException] = []

    def first_search() -> None:
        try:
            client.search("first")
        except BaseException as exc:  # pragma: no cover - assertion aid
            failures.append(exc)

    worker = threading.Thread(target=first_search)
    worker.start()
    assert started.wait(timeout=1)
    try:
        with pytest.raises(OpenLibraryUnavailable, match="already in progress"):
            client.search("second")
    finally:
        release.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert failures == []
    assert len(session.calls) == 1


def test_search_uses_curated_catalogue_when_open_library_is_unavailable():
    service = BookRecommendationService(client=_UnavailableClient())

    payload = service.search("Dune")

    assert payload["source"] == "curated_fallback"
    assert payload["results"][0]["title"] == "Dune"
    assert payload["results"][0]["author_name"] == ["Frank Herbert"]
    assert payload["results"][0]["key"].startswith("/local/books/")
    assert service.search("   ") == {"query": "", "results": [], "source": "none"}


def test_unicode_catalogue_search_does_not_merge_titles_or_match_empty_normalization():
    catalogue = [
        {
            "key": "/local/books/three-body",
            "title": "三体",
            "author_name": ["刘慈欣"],
            "subject": ["科幻小说"],
        },
        {
            "key": "/local/books/wandering-earth",
            "title": "流浪地球",
            "author_name": ["刘慈欣"],
            "subject": ["科幻小说"],
        },
    ]
    service = BookRecommendationService(client=_UnavailableClient(), catalogue=catalogue)

    results = service.search("刘慈欣")["results"]

    assert {book["title"] for book in results} == {"三体", "流浪地球"}
    assert service.search("🌟🌟")["results"] == []


def test_search_normalizes_open_library_work_keys_and_returns_json_data():
    service = BookRecommendationService(client=_StaticClient())

    payload = service.search("Remote Result", limit=1)

    assert payload["results"][0]["key"] == "/works/OL123W"
    assert payload["results"][0]["open_library_url"] == "https://openlibrary.org/works/OL123W"
    json.dumps(payload, allow_nan=False)


def test_search_canonicalizes_languages_and_removes_translated_author_aliases():
    class _AuthorAliasClient:
        def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
            return [
                {
                    "key": "OL777W",
                    "title": "Dune",
                    "author_name": ["Frank Herbert", "Френк Герберт"],
                    # Live Open Library data currently attaches two keys to
                    # these two display aliases even though this is one author.
                    "author_key": ["OL123A", "OL999A"],
                    "language": ["English", "/languages/eng", "zh-CN", "und"],
                },
                {
                    "key": "OL778W",
                    "title": "A Real Collaboration",
                    "author_name": ["Alice Smith", "Bob Jones", "Боб Джонс"],
                    "author_key": ["OL1A", "OL2A"],
                    "language": ["eng"],
                },
            ]

    service = BookRecommendationService(client=_AuthorAliasClient(), catalogue=[])

    dune = service.search("Dune", limit=2)["results"][0]
    collaboration = service.search("Collaboration", limit=2)["results"][1]

    assert dune["author_name"] == ["Frank Herbert"]
    assert dune["language"] == ["eng", "zho"]
    assert collaboration["author_name"] == ["Alice Smith", "Bob Jones"]


def test_recommendations_strictly_match_english_with_noisy_language_metadata():
    catalogue = [
        {
            "key": "/local/books/english",
            "title": "English Candidate",
            "author_name": ["English Author"],
            "subject": ["Science fiction"],
            "language": ["en-US", "English", "und"],
        },
        {
            "key": "/local/books/multilingual",
            "title": "English Translation Available",
            "author_name": ["Another Author"],
            "subject": ["Science fiction"],
            "language": ["chi", "eng"],
        },
        {
            "key": "/local/books/chinese",
            "title": "Chinese Only",
            "author_name": ["中文作者"],
            "subject": ["Science fiction"],
            "language": ["Chinese", "/languages/zho"],
        },
        {
            "key": "/local/books/unknown",
            "title": "Unknown Language",
            "author_name": ["Unknown Author"],
            "subject": ["Science fiction"],
        },
    ]
    service = BookRecommendationService(
        client=_NoNetworkClient(),
        catalogue=catalogue,
        remote_recommendations=False,
    )
    noisy_english_favourite = {
        **DUNE,
        "language": ["English", "/languages/eng", "zh-CN", "und"],
    }

    payload = service.recommend(
        [noisy_english_favourite],
        {"shortlist_size": 6, "theme_list_count": 0},
    )
    books = payload["shortlists"][0]["books"]

    assert {book["title"] for book in books} == {
        "English Candidate",
        "English Translation Available",
    }
    assert all("eng" in book["language"] for book in books)


def test_missing_favourite_language_defaults_to_english_but_known_foreign_does_not():
    catalogue = [
        {
            "key": "/local/books/english-one",
            "title": "English One",
            "author_name": ["One Author"],
            "subject": ["Fantasy"],
            "language": ["eng"],
        },
        {
            "key": "/local/books/chinese-one",
            "title": "中文一",
            "author_name": ["作者一"],
            "subject": ["Fantasy"],
            "language": ["chi"],
        },
    ]
    service = BookRecommendationService(
        client=_NoNetworkClient(),
        catalogue=catalogue,
        remote_recommendations=False,
    )
    favourite = {
        "key": "/works/OLFAVW",
        "title": "Favourite",
        "author_name": ["Favourite Author"],
        "subject": ["Fantasy"],
    }

    defaulted = service.recommend([favourite], {"theme_list_count": 0})
    chinese = service.recommend(
        [{**favourite, "language": ["zh-Hant"]}],
        {"theme_list_count": 0},
    )

    assert [book["title"] for book in defaulted["shortlists"][0]["books"]] == [
        "English One"
    ]
    assert [book["title"] for book in chinese["shortlists"][0]["books"]] == [
        "中文一"
    ]


def test_recommendations_are_deterministic_explainable_and_exclude_inputs():
    service = BookRecommendationService(
        client=_NoNetworkClient(),
        remote_recommendations=False,
    )

    first = service.recommend([DUNE, THE_HOBBIT])
    second = service.recommend([DUNE, THE_HOBBIT])

    assert first == second
    json.dumps(first, allow_nan=False)
    assert first["model"]["version"] == "metadata-content-v1"
    assert first["shortlists"][0]["name"] == "Best overall"
    assert first["shortlists"][0]["kind"] == "overall"
    assert any(item["kind"] == "theme" for item in first["shortlists"][1:])

    input_titles = {"dune", "the hobbit"}
    recommendations = [
        book
        for shortlist in first["shortlists"]
        for book in shortlist["books"]
    ]
    assert recommendations
    assert all(book["title"].casefold() not in input_titles for book in recommendations)
    assert all(book["reasons"] for book in recommendations)
    assert len({book["key"] for book in recommendations}) == len(recommendations)
    assert all(0 <= book["match_score"] <= 100 for book in recommendations)
    assert all(
        {"themes", "style_proxy", "length", "era", "author", "popularity", "language"}
        == set(book["match_factors"])
        for book in recommendations
    )
    assert "metadata-derived proxies" in first["taste_profile"]["style_note"]

    theme_lists = [item for item in first["shortlists"] if item["kind"] == "theme"]
    assert all(item["basis"]["favourite_titles"] for item in theme_lists)
    assert all(item["theme"] in book["matched_themes"] for item in theme_lists for book in item["books"])
    for shortlist in first["shortlists"]:
        first_authors = [
            book["author_name"][0].casefold()
            for book in shortlist["books"]
            if book["author_name"]
        ]
        assert len(first_authors) == len(set(first_authors))


def test_equal_weight_themes_use_genre_salience_instead_of_alphabetical_order():
    service = BookRecommendationService(
        client=_NoNetworkClient(),
        remote_recommendations=False,
    )

    payload = service.recommend([DUNE])

    assert payload["taste_profile"]["themes"][0]["name"] == "Science fiction"
    theme_lists = [item["theme"] for item in payload["shortlists"] if item["kind"] == "theme"]
    assert theme_lists[0] == "Science fiction"


def test_isolated_noisy_subject_does_not_override_repeated_primary_genre():
    noisy_dune = {
        **DUNE,
        "subject": [
            "Fiction, science fiction, general",
            "Science fiction",
            "Science-fiction",
            "American Science fiction",
            "Ecology",
            "Fantasy fiction",
        ],
    }
    service = BookRecommendationService(
        client=_NoNetworkClient(),
        remote_recommendations=False,
    )

    payload = service.recommend([noisy_dune])
    themes = [item["name"] for item in payload["taste_profile"]["themes"]]

    assert "Science fiction" in themes
    assert "Nature & environment" in themes
    assert "Fantasy" not in themes


def test_preferences_filter_themes_and_can_disable_same_author():
    service = BookRecommendationService(
        client=_NoNetworkClient(),
        remote_recommendations=False,
    )

    payload = service.recommend(
        [THE_HOBBIT],
        {
            "exclude_themes": ["Fantasy"],
            "preferred_length": "Medium",
            "preferred_era": "Modern",
            "allow_same_author": False,
            "shortlist_size": 4,
            "theme_list_count": 2,
        },
    )

    books = payload["shortlists"][0]["books"]
    assert len(books) == 4
    assert all("Fantasy" not in book["matched_themes"] for book in books)
    assert all("J. R. R. Tolkien" not in book["author_name"] for book in books)
    assert payload["taste_profile"]["length"]["dominant_band"] == "Medium"

    without_length = service.recommend([THE_HOBBIT], {"ignore_length": True})
    assert without_length["taste_profile"]["length"]["dominant_band"] == "Unknown"
    assert without_length["taste_profile"]["length"]["distribution"] == {}


def test_invalid_favourite_inputs_are_rejected_before_any_client_call():
    service = BookRecommendationService(
        client=_NoNetworkClient(),
        remote_recommendations=False,
    )

    for favourites in (
        [],
        [{"title": "Missing key"}],
        ["not a book"],
        [{"key": {"unexpected": True}, "title": ["Dune"]}],
        [{"key": "https://example.com/book", "title": "Invented"}],
    ):
        try:
            service.recommend(favourites)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError("invalid favourites should raise ValueError")


def test_familiar_discovery_increases_same_author_affinity():
    service = BookRecommendationService(
        client=_NoNetworkClient(),
        remote_recommendations=False,
    )

    balanced = service.recommend([THE_HOBBIT])
    familiar = service.recommend([THE_HOBBIT], {"author_emphasis": 2.0})

    def tolkien_score(payload: dict[str, Any]) -> float:
        return next(
            book["match_score"]
            for book in payload["shortlists"][0]["books"]
            if "J. R. R. Tolkien" in book["author_name"]
        )

    assert tolkien_score(familiar) > tolkien_score(balanced)


def test_each_favourite_contributes_equal_theme_mass_despite_metadata_density():
    sparse_romance = {
        "key": "/works/OL-SPARSE-ROMANCE-W",
        "title": "A Sparse Romance",
        "author_name": ["Example Author"],
        "subject": ["Romance"],
    }
    service = BookRecommendationService(
        client=_NoNetworkClient(),
        remote_recommendations=False,
    )

    payload = service.recommend([DUNE, sparse_romance])
    weights = {
        item["name"]: item["weight"]
        for item in payload["taste_profile"]["themes"]
    }

    assert weights["Romance & relationships"] == pytest.approx(0.5, abs=0.001)
