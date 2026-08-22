from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from .settings import Settings


WHEELO_ROOT = "https://www.wheeloratings.com/src/match_stats/table_data"
SQUIGGLE_ROOT = "https://api.squiggle.com.au/"


TEAM_ALIASES = {
    "adelaide crows": "Adelaide",
    "adelaide": "Adelaide",
    "brisbane": "Brisbane Lions",
    "brisbane lions": "Brisbane Lions",
    "carlton": "Carlton",
    "collingwood": "Collingwood",
    "essendon": "Essendon",
    "fremantle": "Fremantle",
    "geelong": "Geelong",
    "geelong cats": "Geelong",
    "gold coast": "Gold Coast",
    "gold coast suns": "Gold Coast",
    "greater western sydney": "GWS",
    "gws": "GWS",
    "gws giants": "GWS",
    "hawthorn": "Hawthorn",
    "melbourne": "Melbourne",
    "north melbourne": "North Melbourne",
    "kangaroos": "North Melbourne",
    "port adelaide": "Port Adelaide",
    "richmond": "Richmond",
    "st kilda": "St Kilda",
    "sydney": "Sydney",
    "sydney swans": "Sydney",
    "west coast": "West Coast",
    "west coast eagles": "West Coast",
    "western bulldogs": "Western Bulldogs",
}


def canonical_team(value: str | None) -> str:
    if not value:
        return "Unknown"
    cleaned = " ".join(value.replace("&", "and").split()).strip()
    return TEAM_ALIASES.get(cleaned.casefold(), cleaned)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "" or value == "NA":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def columnar_records(data: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Turn the column-oriented JSON used by Wheelo into row dictionaries."""
    if not data:
        return []
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
        else:
            return list(data)

    lengths = [len(value) for value in data.values() if isinstance(value, list)]
    row_count = max(lengths, default=1)
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        row: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, list):
                row[key] = value[index] if index < len(value) else None
            else:
                row[key] = value
        rows.append(row)
    return rows


class PublicDataClient:
    """Small, cached, identifiable client for the public AFL data sources."""

    def __init__(self, settings: Settings, force: bool = False) -> None:
        self.settings = settings
        self.force = force
        self.settings.ensure_directories()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": settings.user_agent,
                "Accept": "application/json",
            }
        )
        self._last_request_at = 0.0

    def _fetch_json(
        self,
        url: str,
        cache_path: Path,
        *,
        mutable: bool = False,
        max_age: timedelta = timedelta(hours=6),
    ) -> dict[str, Any]:
        if cache_path.exists() and not self.force:
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                cache_path.stat().st_mtime,
                tz=timezone.utc,
            )
            if not mutable or age <= max_age:
                return json.loads(cache_path.read_text(encoding="utf-8"))

        delay = 0.08 - (time.monotonic() - self._last_request_at)
        if delay > 0:
            time.sleep(delay)

        error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.session.get(url, timeout=(10, 45))
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                payload = response.json()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(payload, separators=(",", ":")),
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
                return payload
            except (requests.RequestException, ValueError) as exc:
                error = exc
                if attempt < 3:
                    time.sleep(1.5 * (2**attempt))
        raise RuntimeError(f"Unable to retrieve AFL data from {url}: {error}")

    def squiggle_games(self, season: int, *, mutable: bool | None = None) -> list[dict[str, Any]]:
        if mutable is None:
            mutable = season >= self.settings.current_season
        url = f"{SQUIGGLE_ROOT}?q=games;year={season}"
        cache_path = self.settings.raw_dir / "squiggle" / f"games-{season}.json"
        payload = self._fetch_json(url, cache_path, mutable=mutable)
        games = payload.get("games", [])
        return games if isinstance(games, list) else []

    def wheelo_season_index(self, season: int) -> dict[str, Any]:
        url = f"{WHEELO_ROOT}/{season}.json"
        cache_path = self.settings.raw_dir / "wheelo" / f"{season}.json"
        return self._fetch_json(
            url,
            cache_path,
            mutable=season >= self.settings.current_season,
        )

    def wheelo_round(self, round_id: str, *, mutable: bool = False) -> dict[str, Any]:
        url = f"{WHEELO_ROOT}/{round_id}.json"
        cache_path = self.settings.raw_dir / "wheelo" / round_id[:4] / f"{round_id}.json"
        return self._fetch_json(url, cache_path, mutable=mutable)

    def source_manifest(self) -> list[dict[str, Any]]:
        manifest: list[dict[str, Any]] = []
        for path in sorted(self.settings.raw_dir.rglob("*.json")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.append(
                {
                    "path": str(path.relative_to(self.settings.raw_dir)),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                    "retrieved_at": datetime.fromtimestamp(
                        path.stat().st_mtime,
                        tz=timezone.utc,
                    ).isoformat(),
                }
            )
        return manifest


def parse_squiggle_datetime(game: dict[str, Any]) -> datetime:
    raw = str(game.get("date") or "")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raw_offset = str(game.get("tz") or "+10:00")
        sign = -1 if raw_offset.startswith("-") else 1
        hours, minutes = (int(part) for part in raw_offset.lstrip("+-").split(":"))
        parsed = parsed.replace(
            tzinfo=timezone(sign * timedelta(hours=hours, minutes=minutes))
        )
    return parsed


def _fixture_lookup(games: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        pair = (canonical_team(game.get("hteam")), canonical_team(game.get("ateam")))
        try:
            game["_parsed_date"] = parse_squiggle_datetime(game)
        except (TypeError, ValueError):
            continue
        by_pair[pair].append(game)
    for values in by_pair.values():
        values.sort(key=lambda item: item["_parsed_date"])
    return by_pair


def _match_date(season: int, value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.strptime(f"{value} {season}", "%d %b %Y")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone(timedelta(hours=10)))


TEAM_METRICS = (
    "Score",
    "xScore",
    "Inside50s",
    "TotalClearances",
    "CentreClearances",
    "ContestedPossessions",
    "DisposalEfficiency",
    "MetresGained",
    "Tackles",
    "Intercepts",
    "ContestedMarks",
    "ShotsAtGoal",
    "ScoresPerInside50",
    "RatingPoints",
    "Experience",
    "Age",
    "GroundBallGets",
    "PostClearanceContestedPossessions",
    "HitoutsToAdvantage",
    "xChainScore",
    "xChainScoreFromStoppage",
    "xChainScoreFromTurnover",
)


def _summarise_team_stats(team_row: dict[str, Any], players: list[dict[str, Any]]) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        key: _safe_float(team_row.get(key)) for key in TEAM_METRICS
    }
    for output_key, player_key in (
        ("PressureActs", "PressureActs"),
        ("Turnovers", "Turnovers"),
        ("ScoreInvolvements", "ScoreInvolvements"),
        ("PlayerEstimatedRating", "EstimatedRating"),
    ):
        numbers = [_safe_float(player.get(player_key)) for player in players]
        available = [number for number in numbers if number is not None]
        values[output_key] = sum(available) if available else None
    return values


def load_historical_matches(
    client: PublicDataClient,
    start_season: int,
    end_season: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for season in range(start_season, end_season + 1):
        fixtures = client.squiggle_games(season, mutable=season == end_season)
        fixture_map = _fixture_lookup(fixtures)
        season_index = client.wheelo_season_index(season)
        raw_round_ids = season_index.get("RoundId", [])
        round_ids = raw_round_ids if isinstance(raw_round_ids, list) else [raw_round_ids]

        for round_id in round_ids:
            round_id = str(round_id)
            payload = client.wheelo_round(
                round_id,
                mutable=season == end_season and round_id == str(round_ids[-1]),
            )
            if not payload.get("Matches") or not payload.get("TeamData"):
                continue
            summary_rows = columnar_records(payload.get("Summary"))
            summary = summary_rows[0] if summary_rows else {}
            match_rows = columnar_records(payload.get("Matches"))
            team_rows = columnar_records(payload.get("TeamData"))
            player_rows = columnar_records(payload.get("Data"))

            teams_by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in team_rows:
                teams_by_match[str(row.get("MatchId"))].append(row)
            players_by_match_team: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in player_rows:
                key = (str(row.get("MatchId")), canonical_team(row.get("Team")))
                players_by_match_team[key].append(row)

            for match_row in match_rows:
                match_id = str(match_row.get("MatchId"))
                home_team = canonical_team(match_row.get("HomeTeam"))
                away_team = canonical_team(match_row.get("AwayTeam"))
                team_candidates = teams_by_match.get(match_id, [])
                home_row = next(
                    (
                        row
                        for row in team_candidates
                        if canonical_team(row.get("Team")) == home_team
                        or row.get("Abbreviation") == match_row.get("HomeAbbreviation")
                    ),
                    {},
                )
                away_row = next(
                    (
                        row
                        for row in team_candidates
                        if canonical_team(row.get("Team")) == away_team
                        or row.get("Abbreviation") == match_row.get("AwayAbbreviation")
                    ),
                    {},
                )
                if not home_row or not away_row:
                    continue

                approximate_date = _match_date(season, match_row.get("MatchDate"))
                candidates = fixture_map.get((home_team, away_team), [])
                fixture: dict[str, Any] | None = None
                if candidates:
                    if approximate_date:
                        fixture = min(
                            candidates,
                            key=lambda item: abs(
                                (item["_parsed_date"].date() - approximate_date.date()).days
                            ),
                        )
                    else:
                        fixture = candidates[0]
                    candidates.remove(fixture)

                start_time = fixture.get("_parsed_date") if fixture else approximate_date
                if start_time is None:
                    start_time = datetime(season, 1, 1, tzinfo=timezone.utc)
                home_players = players_by_match_team.get((match_id, home_team), [])
                away_players = players_by_match_team.get((match_id, away_team), [])
                home_score = _safe_float(home_row.get("Score"))
                away_score = _safe_float(away_row.get("Score"))
                if home_score is None or away_score is None:
                    continue

                matches.append(
                    {
                        "match_id": match_id,
                        "source_game_id": fixture.get("id") if fixture else None,
                        "season": season,
                        "round_number": int(summary.get("RoundNumber") or 0),
                        "round_name": summary.get("RoundName") or f"Round {summary.get('RoundNumber', '')}",
                        "start_time": start_time,
                        "venue": fixture.get("venue") if fixture else "Unknown venue",
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_stats": _summarise_team_stats(home_row, home_players),
                        "away_stats": _summarise_team_stats(away_row, away_players),
                        "home_players": home_players,
                        "away_players": away_players,
                        "is_final": bool(fixture.get("is_final")) if fixture else False,
                    }
                )

    matches.sort(key=lambda match: (match["start_time"], match["match_id"]))
    return matches


def upcoming_round(games: Iterable[dict[str, Any]], now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    candidates: list[dict[str, Any]] = []
    for original in games:
        game = dict(original)
        try:
            start_time = parse_squiggle_datetime(game)
        except (TypeError, ValueError):
            continue
        is_complete = int(game.get("complete") or 0) >= 100
        if not is_complete and start_time >= now - timedelta(hours=4):
            game["start_time"] = start_time
            candidates.append(game)
    if not candidates:
        return []
    candidates.sort(key=lambda game: game["start_time"])
    first_round = candidates[0].get("round")
    round_games = [game for game in candidates if game.get("round") == first_round]
    return sorted(round_games, key=lambda game: game["start_time"])
