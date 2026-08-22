from __future__ import annotations

import hmac
import os
import threading
import time
from datetime import datetime
from functools import lru_cache
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, render_template, request

from afl_ml.artifacts import load_json
from afl_ml.settings import Settings

try:
    from dotenv import load_dotenv
except ImportError:  # App Service settings are already provided as environment variables.
    def load_dotenv() -> bool:
        return False


load_dotenv()

MELBOURNE = ZoneInfo("Australia/Melbourne")
REFRESH_LOCK = threading.Lock()

TEAM_COLOURS = {
    "Adelaide": ("#0b2341", "#f6c343"),
    "Brisbane Lions": ("#7c1734", "#f2b544"),
    "Carlton": ("#102a43", "#e9f2f8"),
    "Collingwood": ("#111111", "#f5f5f2"),
    "Essendon": ("#171717", "#df2e38"),
    "Fremantle": ("#37225f", "#f4f2f7"),
    "Geelong": ("#142a4a", "#f4f4ee"),
    "Gold Coast": ("#d9292f", "#f7c840"),
    "GWS": ("#e26b20", "#222222"),
    "Hawthorn": ("#4b2b25", "#f5b942"),
    "Melbourne": ("#152b4e", "#d92c3a"),
    "North Melbourne": ("#1f5ba8", "#f4f7fb"),
    "Port Adelaide": ("#111111", "#27a7ad"),
    "Richmond": ("#161616", "#f4c542"),
    "St Kilda": ("#d72638", "#161616"),
    "Sydney": ("#d9292f", "#f6f3eb"),
    "West Coast": ("#123b76", "#f2c84b"),
    "Western Bulldogs": ("#1f53a0", "#d9283d"),
}

INDICATOR_EXPLANATIONS = {
    "form_margin_long_diff": (
        "Scoring margin from earlier matches, updated slowly so roughly the last "
        "five to ten games describe sustained form rather than one result."
    ),
    "elo_diff": (
        "An Elo-style rating that starts each club near 1500, adjusts for opponent "
        "strength and result margin, and moves one-third toward average each season."
    ),
    "shots_short_diff": (
        "Recent goals plus behinds. It measures how often a team creates a scoring "
        "opportunity, regardless of whether those shots were converted."
    ),
    "xdefence_short_diff": (
        "The expected score conceded from recent opposition chances. Lower values "
        "indicate that a team is allowing fewer or lower-quality opportunities."
    ),
    "player_rating_diff": (
        "Rolling AFL player-rating output for each club's most recent lineup, with "
        "limited samples pulled toward a player's established rating."
    ),
    "inside50_long_diff": (
        "Forward-50 entries across prior matches, updated slowly to capture sustained "
        "territory and chance creation."
    ),
}


def _book_length_label(pages: object) -> str | None:
    try:
        count = int(float(pages))
    except (TypeError, ValueError, OverflowError):
        return None
    if count < 250:
        return "Short"
    if count < 400:
        return "Medium"
    if count < 600:
        return "Long"
    return "Epic"


def _book_for_browser(book: dict) -> dict:
    authors = [str(value) for value in book.get("author_name", []) if value]
    key = str(book.get("key") or "")
    if key.startswith("/works/"):
        open_library_url = f"https://openlibrary.org{key}"
    else:
        search_text = " ".join(
            value for value in [str(book.get("title") or ""), authors[0] if authors else ""] if value
        )
        open_library_url = f"https://openlibrary.org/search?q={quote_plus(search_text)}"
    return {
        **book,
        "authors": authors,
        "author": ", ".join(authors) or "Unknown author",
        "year": book.get("first_publish_year"),
        "open_library_url": open_library_url,
        "length_label": book.get("length_band")
        or _book_length_label(book.get("number_of_pages_median")),
        "themes": list(book.get("matched_themes") or []),
    }


def _recommendations_for_browser(payload: dict) -> dict:
    profile = payload.get("taste_profile", {})
    themes = [item.get("name") for item in profile.get("themes", []) if item.get("name")]
    styles = [
        item.get("name")
        for item in profile.get("style_proxies", [])
        if item.get("name")
    ]
    if themes:
        theme_phrase = " and ".join(themes[:2])
        summary = f"Themes: {theme_phrase}."
    else:
        summary = "Mixed themes."
    if styles:
        summary += f" Style: {styles[0]}."

    lists = []
    for index, shortlist in enumerate(payload.get("shortlists", [])):
        basis = shortlist.get("basis")
        if isinstance(basis, dict):
            titles = [str(value) for value in basis.get("favourite_titles", []) if value]
            description = (
                f"Based on {', '.join(titles[:3])}."
                if titles
                else "Matches one of your main themes."
            )
        else:
            description = "Closest overall matches." if index == 0 else str(
                basis or "Matches your selected books."
            )
        lists.append(
            {
                "id": f"shortlist-{index + 1}",
                "title": shortlist.get("name", "Reading direction"),
                "description": description,
                "books": [_book_for_browser(book) for book in shortlist.get("books", [])],
            }
        )

    return {
        "profile": {
            "themes": themes,
            "styles": styles,
            "summary": summary,
            "details": profile,
        },
        "lists": lists,
        "meta": {
            **payload.get("model", {}),
            "notices": payload.get("notices", []),
        },
    }


def _team_initials(team: str) -> str:
    special = {
        "Brisbane Lions": "BL",
        "Gold Coast": "GC",
        "North Melbourne": "NM",
        "Port Adelaide": "PA",
        "St Kilda": "SK",
        "West Coast": "WC",
        "Western Bulldogs": "WB",
    }
    if team in special:
        return special[team]
    return "".join(word[0] for word in team.split()[:2]).upper()


def _display_time(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(MELBOURNE).strftime("%a %-d %b · %-I:%M %p")


def _margin_phrase(value: float, home_team: str, away_team: str) -> str:
    if abs(float(value)) < 0.5:
        return "Level"
    winner = home_team if float(value) > 0 else away_team
    return f"{winner} by {abs(float(value)):.0f}"


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings()
    app = Flask(__name__)
    app.config["SETTINGS"] = settings
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    book_service_lock = threading.Lock()

    @lru_cache(maxsize=1)
    def model_report() -> dict:
        return load_json(settings.report_path, default={})

    @lru_cache(maxsize=2)
    def _prediction_payload_cached(_minute_bucket: int) -> tuple[dict, str]:
        # Public page views use the local snapshot by default. This prevents a
        # visitor (or platform probe) from starting a serverless SQL billing
        # window. Scheduled refreshes still persist predictions to Azure SQL.
        if (
            settings.database_read_enabled
            and settings.database_enabled
            and settings.db_server
        ):
            try:
                from afl_ml.database import load_predictions

                stored = load_predictions(settings)
                if stored:
                    return stored, "Azure SQL"
            except Exception:
                app.logger.warning("Azure SQL unavailable; serving the bundled snapshot")
        return load_json(
            settings.predictions_path,
            default={"round_name": "Predictions pending", "predictions": []},
        ), "local model snapshot"

    def prediction_payload() -> tuple[dict, str]:
        return _prediction_payload_cached(int(time.monotonic() // 60))

    def book_recommendation_service():
        service = app.extensions.get("book_recommendation_service")
        if service is None:
            with book_service_lock:
                service = app.extensions.get("book_recommendation_service")
                if service is None:
                    from book_recommender import BookRecommendationService

                    service = BookRecommendationService()
                    app.extensions["book_recommendation_service"] = service
        return service

    @app.context_processor
    def template_helpers() -> dict:
        return {
            "team_initials": _team_initials,
            "team_colours": lambda team: TEAM_COLOURS.get(team, ("#26364a", "#f3f6f8")),
            "display_time": _display_time,
            "margin_phrase": _margin_phrase,
            "indicator_explanation": lambda feature: INDICATOR_EXPLANATIONS.get(
                feature,
                "A pre-match team difference calculated only from earlier games.",
            ),
        }

    @app.get("/")
    def index():
        return render_template("landing.html")

    @app.get("/afl")
    def afl_predictions():
        payload, storage_source = prediction_payload()
        return render_template(
            "index.html",
            payload=payload,
            predictions=payload.get("predictions", []),
            report=model_report(),
            storage_source=storage_source,
        )

    @app.get("/books")
    def books():
        return render_template("books.html")

    @app.get("/api/books/search")
    def api_book_search():
        query = " ".join(request.args.get("q", "").split())
        if len(query) < 2:
            return jsonify({"error": "Enter at least two characters to search for a book."}), 400
        if len(query) > 120:
            return jsonify({"error": "That search is too long."}), 400
        result = book_recommendation_service().search(query, limit=8)
        books = [_book_for_browser(book) for book in result.get("results", [])]
        if result.get("degraded") and not books:
            return jsonify(
                {"error": "Book search is temporarily unavailable. Please try again."}
            ), 503
        return jsonify(
            {
                "books": books,
                "source": result.get("source"),
                "degraded": bool(result.get("degraded")),
            }
        )

    @app.post("/api/books/recommend")
    def api_book_recommendations():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Send a JSON object with a favourites list."}), 400
        favourites = body.get("favourites")
        if not isinstance(favourites, list) or not 1 <= len(favourites) <= 10:
            return jsonify({"error": "Choose between one and ten favourite books."}), 400

        browser_preferences = body.get("preferences")
        if browser_preferences is None:
            browser_preferences = {}
        elif not isinstance(browser_preferences, dict):
            return jsonify({"error": "Preferences must be a JSON object."}), 400
        discovery = browser_preferences.get("discovery")
        service_preferences = {
            "shortlist_size": 6,
            "theme_list_count": 3,
            "allow_same_author": discovery != "adventurous",
            "author_emphasis": 2.0 if discovery == "familiar" else 1.0,
            "ignore_length": browser_preferences.get("length") == "any",
        }
        era = browser_preferences.get("era")
        if era == "modern":
            service_preferences["preferred_era"] = ["Modern", "Contemporary"]
        elif era == "classics":
            service_preferences["preferred_era"] = ["Classic", "Post-war"]

        try:
            result = book_recommendation_service().recommend(
                favourites,
                service_preferences,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(_recommendations_for_browser(result))

    @app.get("/api/books/model")
    def api_book_model():
        return jsonify(
            {
                "name": "Explainable metadata content model",
                "version": "metadata-content-v1",
                "data_source": "Open Library with a curated resilience catalogue",
                "signals": [
                    "themes and detailed subjects",
                    "metadata-derived style proxies",
                    "author",
                    "length band",
                    "publication era",
                    "language",
                    "reader interest and rating evidence",
                ],
                "method": "Weighted content similarity with popularity as a bounded quality prior and diversity-aware shortlists.",
                "limitations": [
                    "The model does not analyse full copyrighted text.",
                    "Style labels are proxies derived from available metadata.",
                    "It has no personal feedback history yet, so results are suggestions rather than calibrated predictions.",
                ],
            }
        )

    @app.get("/api/predictions")
    def api_predictions():
        payload, _ = prediction_payload()
        return jsonify(payload)

    @app.get("/api/model")
    def api_model():
        return jsonify(model_report())

    @app.get("/health")
    def health():
        # App Service probes must stay fast and must not keep a serverless SQL
        # database awake. SQL connectivity has its own explicit readiness route.
        payload = load_json(
            settings.predictions_path,
            default={"predictions": []},
        )
        return jsonify(
            {
                "status": "ok",
                "model_version": payload.get("model_version"),
                "predictions": len(payload.get("predictions", [])),
                "database_configured": bool(settings.db_server and settings.db_name),
            }
        )

    @app.get("/ready")
    def ready():
        if not settings.db_server:
            return jsonify({"status": "ready", "database": "snapshot mode"})
        from afl_ml.database import database_health

        ok, detail = database_health(settings)
        if not ok:
            app.logger.warning("Azure SQL readiness check failed: %s", detail)
        status = 200 if ok else 503
        database_state = "connected" if ok else "unavailable"
        return jsonify({"status": "ready" if ok else "degraded", "database": database_state}), status

    @app.post("/api/admin/refresh")
    def admin_refresh():
        configured = settings.refresh_token
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not configured:
            return jsonify({"error": "Prediction refresh is not configured"}), 503
        if not supplied or not hmac.compare_digest(configured, supplied):
            return jsonify({"error": "Unauthorized"}), 401
        if not REFRESH_LOCK.acquire(blocking=False):
            return jsonify({"error": "A refresh is already running"}), 409
        try:
            # The full pandas/scikit-learn stack is needed only by the scheduled
            # refresh, not for normal web requests or container health probes.
            from afl_ml.service import refresh_prediction_snapshot

            result = refresh_prediction_snapshot(
                settings,
                persist_db=True,
                force=True,
            )
            _prediction_payload_cached.cache_clear()
            return jsonify(
                {
                    "status": "refreshed",
                    "round_name": result.get("round_name"),
                    "predictions": len(result.get("predictions", [])),
                    "state_updates_applied": result.get("state_updates_applied", 0),
                }
            )
        finally:
            REFRESH_LOCK.release()

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
