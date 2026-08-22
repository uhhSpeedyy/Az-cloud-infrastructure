from __future__ import annotations

from dataclasses import replace

from afl_ml.settings import Settings
from app import create_app


class _FakeBookService:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []
        self.recommend_calls: list[tuple[list[dict], dict]] = []

    def search(self, query: str, limit: int = 8) -> dict:
        self.search_calls.append((query, limit))
        return {
            "source": "fixture",
            "results": [
                {
                    "key": "/works/OL893414W",
                    "title": "Dune",
                    "author_name": ["Frank Herbert"],
                    "first_publish_year": 1965,
                    "number_of_pages_median": 606,
                    "first_sentence": ["A beginning is the time for taking care."],
                    "readinglog_count": 100,
                    "want_to_read_count": 60,
                    "currently_reading_count": 5,
                    "already_read_count": 35,
                    "cover_url": "https://covers.openlibrary.org/b/id/1-M.jpg",
                }
            ],
        }

    def recommend(self, favourites: list[dict], preferences: dict) -> dict:
        self.recommend_calls.append((favourites, preferences))
        return {
            "model": {"name": "Fixture model", "version": "test-v1"},
            "taste_profile": {
                "themes": [{"name": "Science fiction", "weight": 1.0}],
                "style_proxies": [{"name": "Epic and expansive", "weight": 1.0}],
            },
            "shortlists": [
                {
                    "name": "Best overall",
                    "basis": "Balanced similarity.",
                    "books": [
                        {
                            "key": "/works/OL27448W",
                            "title": "The Lord of the Rings",
                            "author_name": ["J. R. R. Tolkien"],
                            "first_publish_year": 1954,
                            "length_band": "Epic",
                            "matched_themes": ["Fantasy"],
                            "reasons": ["Shares your interest in epic stories."],
                        }
                    ],
                }
            ],
            "notices": ["Fixture notice"],
        }


def _app(tmp_path):
    settings = replace(
        Settings(),
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        database_enabled=False,
    )
    settings.ensure_directories()
    app = create_app(settings)
    app.config["TESTING"] = True
    return app


def test_book_page_and_model_card_do_not_initialise_the_service(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()

    page = client.get("/books")
    assert page.status_code == 200
    assert b"Add books you like" in page.data
    assert b"Recommend books" in page.data

    model = client.get("/api/books/model")
    assert model.status_code == 200
    assert model.get_json()["version"] == "metadata-content-v1"

    health = client.get("/health")
    assert health.status_code == 200
    assert "book_recommendation_service" not in app.extensions


def test_book_search_validates_and_adapts_results_for_the_browser(tmp_path):
    app = _app(tmp_path)
    service = _FakeBookService()
    app.extensions["book_recommendation_service"] = service
    client = app.test_client()

    invalid = client.get("/api/books/search?q=a")
    assert invalid.status_code == 400

    response = client.get("/api/books/search?q=%20Dune%20")
    assert response.status_code == 200
    assert service.search_calls == [("Dune", 8)]
    book = response.get_json()["books"][0]
    assert book["authors"] == ["Frank Herbert"]
    assert book["year"] == 1965
    assert book["length_label"] == "Epic"
    assert book["open_library_url"] == "https://openlibrary.org/works/OL893414W"
    assert book["first_sentence"] == ["A beginning is the time for taking care."]
    assert book["readinglog_count"] == 100


def test_book_recommendations_map_preferences_and_return_shortlists(tmp_path):
    app = _app(tmp_path)
    service = _FakeBookService()
    app.extensions["book_recommendation_service"] = service
    client = app.test_client()
    favourite = {
        "key": "/works/OL893414W",
        "title": "Dune",
        "author_name": ["Frank Herbert"],
    }

    response = client.post(
        "/api/books/recommend",
        json={
            "favourites": [favourite],
            "preferences": {
                "era": "modern",
                "length": "any",
                "discovery": "adventurous",
            },
        },
    )

    assert response.status_code == 200
    assert service.recommend_calls[0][0] == [favourite]
    mapped = service.recommend_calls[0][1]
    assert mapped["preferred_era"] == ["Modern", "Contemporary"]
    assert mapped["ignore_length"] is True
    assert mapped["allow_same_author"] is False
    assert mapped["author_emphasis"] == 1.0
    payload = response.get_json()
    assert payload["profile"]["themes"] == ["Science fiction"]
    assert payload["lists"][0]["title"] == "Best overall"
    assert payload["lists"][0]["books"][0]["length_label"] == "Epic"

    familiar = client.post(
        "/api/books/recommend",
        json={"favourites": [favourite], "preferences": {"discovery": "familiar"}},
    )
    assert familiar.status_code == 200
    assert service.recommend_calls[1][1]["author_emphasis"] == 2.0


def test_book_search_reports_an_upstream_outage_when_no_fallback_matches(tmp_path):
    app = _app(tmp_path)

    class _DegradedService(_FakeBookService):
        def search(self, query: str, limit: int = 8) -> dict:
            return {"source": "curated_fallback", "results": [], "degraded": True}

    app.extensions["book_recommendation_service"] = _DegradedService()
    response = app.test_client().get("/api/books/search?q=Unknown%20Title")

    assert response.status_code == 503
    assert "temporarily unavailable" in response.get_json()["error"]


def test_book_recommendations_reject_invalid_payloads(tmp_path):
    client = _app(tmp_path).test_client()

    assert client.post("/api/books/recommend", data="not json").status_code == 400
    assert client.post("/api/books/recommend", json={"favourites": []}).status_code == 400
    assert (
        client.post(
            "/api/books/recommend",
            json={"favourites": [{"key": "/works/1", "title": "A"}], "preferences": []},
        ).status_code
        == 400
    )


def test_book_recommendations_reject_oversized_requests(tmp_path):
    client = _app(tmp_path).test_client()
    oversized_json = b'{"favourites":[{"key":"/works/1","title":"' + (b"a" * 70_000) + b'"}]}'

    response = client.post(
        "/api/books/recommend",
        data=oversized_json,
        content_type="application/json",
    )

    assert response.status_code == 413
