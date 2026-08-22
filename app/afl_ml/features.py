from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_sources import canonical_team


TEAM_REGION = {
    "Adelaide": "SA",
    "Port Adelaide": "SA",
    "Brisbane Lions": "QLD",
    "Gold Coast": "QLD",
    "Sydney": "NSW",
    "GWS": "NSW",
    "Fremantle": "WA",
    "West Coast": "WA",
    "Carlton": "VIC",
    "Collingwood": "VIC",
    "Essendon": "VIC",
    "Geelong": "VIC",
    "Hawthorn": "VIC",
    "Melbourne": "VIC",
    "North Melbourne": "VIC",
    "Richmond": "VIC",
    "St Kilda": "VIC",
    "Western Bulldogs": "VIC",
}

VENUE_REGION = {
    "M.C.G.": "VIC",
    "Docklands": "VIC",
    "Marvel Stadium": "VIC",
    "Kardinia Park": "VIC",
    "GMHBA Stadium": "VIC",
    "Princes Park": "VIC",
    "Eureka Stadium": "VIC",
    "Mars Stadium": "VIC",
    "Adelaide Oval": "SA",
    "Football Park": "SA",
    "Norwood Oval": "SA",
    "Barossa Park": "SA",
    "Adelaide Hills": "SA",
    "Perth Stadium": "WA",
    "Optus Stadium": "WA",
    "Subiaco": "WA",
    "Hands Oval": "WA",
    "Gabba": "QLD",
    "Carrara": "QLD",
    "Cazaly's Stadium": "QLD",
    "Riverway Stadium": "QLD",
    "S.C.G.": "NSW",
    "Sydney Showground": "NSW",
    "Stadium Australia": "NSW",
    "Blacktown": "NSW",
    "Manuka Oval": "ACT",
    "UNSW Canberra Oval": "ACT",
    "York Park": "TAS",
    "University of Tasmania Stadium": "TAS",
    "Bellerive Oval": "TAS",
    "Marrara Oval": "NT",
    "Traeger Park": "NT",
    "Wellington": "NZ",
    "Jiangwan Stadium": "CHN",
    "Adelaide Arena at Jiangwan Stadium": "CHN",
}

REGION_DISTANCE_KM = {
    frozenset(("VIC", "NSW")): 720,
    frozenset(("VIC", "ACT")): 660,
    frozenset(("VIC", "SA")): 730,
    frozenset(("VIC", "QLD")): 1_370,
    frozenset(("VIC", "WA")): 2_720,
    frozenset(("VIC", "TAS")): 600,
    frozenset(("NSW", "QLD")): 920,
    frozenset(("NSW", "ACT")): 290,
    frozenset(("NSW", "SA")): 1_160,
    frozenset(("NSW", "WA")): 3_290,
    frozenset(("NSW", "TAS")): 1_060,
    frozenset(("ACT", "QLD")): 1_180,
    frozenset(("ACT", "SA")): 1_160,
    frozenset(("ACT", "WA")): 3_090,
    frozenset(("ACT", "TAS")): 850,
    frozenset(("SA", "QLD")): 1_600,
    frozenset(("SA", "WA")): 2_130,
    frozenset(("SA", "TAS")): 1_160,
    frozenset(("QLD", "WA")): 3_620,
    frozenset(("QLD", "TAS")): 1_790,
    frozenset(("WA", "TAS")): 3_000,
    frozenset(("VIC", "NT")): 2_250,
    frozenset(("NSW", "NT")): 3_000,
    frozenset(("ACT", "NT")): 3_100,
    frozenset(("SA", "NT")): 1_550,
    frozenset(("QLD", "NT")): 2_850,
    frozenset(("WA", "NT")): 2_650,
    frozenset(("TAS", "NT")): 3_300,
    frozenset(("VIC", "NZ")): 2_600,
    frozenset(("NSW", "NZ")): 2_250,
    frozenset(("ACT", "NZ")): 2_330,
    frozenset(("SA", "NZ")): 3_200,
    frozenset(("QLD", "NZ")): 2_500,
    frozenset(("WA", "NZ")): 5_250,
    frozenset(("TAS", "NZ")): 2_400,
    frozenset(("VIC", "CHN")): 8_050,
    frozenset(("NSW", "CHN")): 7_900,
    frozenset(("ACT", "CHN")): 8_100,
    frozenset(("SA", "CHN")): 7_300,
    frozenset(("QLD", "CHN")): 7_300,
    frozenset(("WA", "CHN")): 7_050,
    frozenset(("TAS", "CHN")): 8_500,
}


FEATURE_INFO: dict[str, dict[str, Any]] = {
    "elo_diff": {
        "label": "Team strength rating",
        "unit": "rating pts",
        "description": "Opponent-adjusted results rating before the match.",
    },
    "form_margin_short_diff": {
        "label": "Recent margin form",
        "unit": "pts/game",
        "description": "Exponentially weighted scoring margin from recent matches.",
    },
    "form_margin_long_diff": {
        "label": "Longer-term margin form",
        "unit": "pts/game",
        "description": "A slower-moving view of scoring margin over prior matches.",
    },
    "attack_short_diff": {
        "label": "Recent scoring",
        "unit": "pts/game",
        "description": "Points scored in prior games, weighted toward recent matches.",
    },
    "defence_short_diff": {
        "label": "Recent defence",
        "unit": "pts allowed/game",
        "description": "Points conceded in prior games; lower is stronger.",
        "lower_is_better": True,
    },
    "xattack_short_diff": {
        "label": "Expected-score attack",
        "unit": "xPts/game",
        "description": "Quality and volume of recent scoring chances before this match.",
    },
    "xdefence_short_diff": {
        "label": "Expected-score defence",
        "unit": "xPts allowed/game",
        "description": "Expected score conceded in prior games; lower is stronger.",
        "lower_is_better": True,
    },
    "finishing_short_diff": {
        "label": "Scoring above expectation",
        "unit": "pts/game",
        "description": "Actual score minus xScore, with recent games weighted most.",
    },
    "inside50_short_diff": {
        "label": "Inside 50s",
        "unit": "/game",
        "description": "Recent forward-50 entries, a measure of chance creation.",
    },
    "inside50_long_diff": {
        "label": "Inside 50s, longer term",
        "unit": "/game",
        "description": "Slower-moving forward-50 entry form.",
    },
    "clearance_short_diff": {
        "label": "Clearances",
        "unit": "/game",
        "description": "Recent total clearances before this match.",
    },
    "centre_clearance_short_diff": {
        "label": "Centre clearances",
        "unit": "/game",
        "description": "Recent centre-bounce clearances before this match.",
    },
    "contested_short_diff": {
        "label": "Contested possessions",
        "unit": "/game",
        "description": "Recent contested-ball wins before this match.",
    },
    "ground_ball_short_diff": {
        "label": "Ground-ball gets",
        "unit": "/game",
        "description": "Recent ability to win the ball at ground level.",
    },
    "post_clearance_short_diff": {
        "label": "Post-clearance contested ball",
        "unit": "/game",
        "description": "Contested possessions won after the initial clearance phase.",
    },
    "disposal_efficiency_short_diff": {
        "label": "Disposal efficiency",
        "unit": "%",
        "description": "Recent effective disposal percentage.",
    },
    "metres_gained_short_diff": {
        "label": "Metres gained",
        "unit": "m/game",
        "description": "Recent territory gained with ball movement.",
    },
    "pressure_short_diff": {
        "label": "Pressure acts",
        "unit": "/game",
        "description": "Recent team pressure, summed from player match statistics.",
    },
    "turnover_short_diff": {
        "label": "Ball security",
        "unit": "turnovers/game",
        "description": "Recent turnovers; lower is stronger.",
        "lower_is_better": True,
    },
    "intercept_short_diff": {
        "label": "Intercept possessions",
        "unit": "/game",
        "description": "Recent possessions won from opposition ball movement.",
    },
    "contested_marks_short_diff": {
        "label": "Contested marks",
        "unit": "/game",
        "description": "Recent marks won in a contest.",
    },
    "shots_short_diff": {
        "label": "Scoring shots",
        "unit": "/game",
        "description": "Recent shots at goal before this match.",
    },
    "team_rating_short_diff": {
        "label": "Team AFL Player Rating",
        "unit": "rating pts/game",
        "description": "Recent sum of official-style player rating output.",
    },
    "player_rating_diff": {
        "label": "Estimated player strength",
        "unit": "rating pts",
        "description": "Pre-match rolling ratings for the club's most recent lineup.",
    },
    "player_top6_diff": {
        "label": "Top-six player strength",
        "unit": "rating pts",
        "description": "Average rolling rating of the six strongest recent players.",
    },
    "lineup_continuity_diff": {
        "label": "Lineup continuity",
        "unit": "%",
        "description": "Share of players retained across the two prior matches.",
    },
    "experience_short_diff": {
        "label": "Team experience",
        "unit": "games/player",
        "description": "Average match experience of recent teams.",
    },
    "rest_days_diff": {
        "label": "Rest advantage",
        "unit": "days",
        "description": "Difference in days since each club's prior match.",
    },
    "travel_advantage_km": {
        "label": "Travel advantage",
        "unit": "km",
        "description": "Approximate extra travel required of the away team.",
    },
    "venue_familiarity_diff": {
        "label": "Venue familiarity",
        "unit": "log games",
        "description": "Difference in prior matches played at this venue.",
    },
    "home_state_advantage": {
        "label": "Home-state advantage",
        "unit": "indicator",
        "description": "Whether the venue is in one club's home region but not the other's.",
    },
    "finals_match": {
        "label": "Finals match",
        "unit": "indicator",
        "description": "Allows for a different scoring environment in finals.",
    },
    "season_progress": {
        "label": "Season progress",
        "unit": "share",
        "description": "Where the match sits within the season.",
    },
    "covid_2020": {
        "label": "2020 match conditions",
        "unit": "indicator",
        "description": "Controls for shortened quarters and the 2020 fixture regime.",
    },
    "prior_games_min": {
        "label": "Form sample size",
        "unit": "games",
        "description": "Prior-match sample available for the less-established club.",
    },
}

FEATURE_COLUMNS = list(FEATURE_INFO)


FORM_KEYS = {
    "form_margin": "margin",
    "attack": "score_for",
    "defence": "score_against",
    "xattack": "xscore_for",
    "xdefence": "xscore_against",
    "finishing": "finishing",
    "inside50": "inside50",
    "clearance": "clearance",
    "centre_clearance": "centre_clearance",
    "contested": "contested",
    "ground_ball": "ground_ball",
    "post_clearance": "post_clearance",
    "disposal_efficiency": "disposal_efficiency",
    "metres_gained": "metres_gained",
    "pressure": "pressure",
    "turnover": "turnover",
    "intercept": "intercept",
    "contested_marks": "contested_marks",
    "shots": "shots",
    "team_rating": "team_rating",
    "experience": "experience",
}

LONG_FORM_KEYS = {
    "form_margin",
    "inside50",
}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _travel_distance(team: str, venue: str) -> float:
    home_region = TEAM_REGION.get(team)
    venue_region = VENUE_REGION.get(venue)
    if not home_region or not venue_region or home_region == venue_region:
        return 0.0
    return float(REGION_DISTANCE_KM.get(frozenset((home_region, venue_region)), 900))


def _player_key(player: dict[str, Any]) -> str:
    value = player.get("WebsiteId") or player.get("PlayerId") or player.get("Player")
    return str(value)


@dataclass
class TeamState:
    short: dict[str, float] = field(default_factory=dict)
    long: dict[str, float] = field(default_factory=dict)
    games: int = 0
    elo: float = 1500.0
    last_date: datetime | None = None
    last_lineup: list[str] = field(default_factory=list)
    continuity: float | None = None


@dataclass
class PlayerState:
    rating: float = 8.0
    games: int = 0


class FeatureEngine:
    def __init__(self) -> None:
        self.teams: dict[str, TeamState] = defaultdict(TeamState)
        self.players: dict[tuple[str, str], PlayerState] = defaultdict(PlayerState)
        self.venue_games: dict[tuple[str, str], int] = defaultdict(int)
        self.processed_match_ids: set[str] = set()
        self.current_season: int | None = None

    def copy(self) -> "FeatureEngine":
        return deepcopy(self)

    def begin_season(self, season: int) -> None:
        if self.current_season == season:
            return
        if self.current_season is not None:
            for state in self.teams.values():
                state.elo = 1500.0 + 0.67 * (state.elo - 1500.0)
        self.current_season = season

    def _lineup_summary(self, team: str) -> tuple[float | None, float | None, float | None]:
        state = self.teams[team]
        if not state.last_lineup:
            return None, None, state.continuity
        ratings = [self.players[(team, player)].rating for player in state.last_lineup]
        if not ratings:
            return None, None, state.continuity
        ratings.sort(reverse=True)
        return float(np.mean(ratings)), float(np.mean(ratings[:6])), state.continuity

    @staticmethod
    def _state_value(state: TeamState, name: str, window: str = "short") -> float | None:
        values = state.short if window == "short" else state.long
        return values.get(name)

    def features_for_fixture(self, fixture: dict[str, Any]) -> dict[str, Any]:
        season = int(fixture["season"])
        self.begin_season(season)
        home_team = canonical_team(fixture["home_team"])
        away_team = canonical_team(fixture["away_team"])
        venue = str(fixture.get("venue") or "Unknown venue")
        start_time: datetime = fixture["start_time"]
        home = self.teams[home_team]
        away = self.teams[away_team]
        row: dict[str, Any] = {}

        def add_difference(
            feature: str,
            home_value: float | None,
            away_value: float | None,
        ) -> None:
            if home_value is None or away_value is None:
                row[feature] = np.nan
            else:
                row[feature] = home_value - away_value
            row[f"home__{feature}"] = home_value
            row[f"away__{feature}"] = away_value

        add_difference("elo_diff", home.elo, away.elo)
        for label, state_key in FORM_KEYS.items():
            feature = f"{label}_short_diff"
            add_difference(
                feature,
                self._state_value(home, state_key, "short"),
                self._state_value(away, state_key, "short"),
            )
            if label in LONG_FORM_KEYS:
                feature = f"{label}_long_diff"
                add_difference(
                    feature,
                    self._state_value(home, state_key, "long"),
                    self._state_value(away, state_key, "long"),
                )

        home_player, home_top6, home_continuity = self._lineup_summary(home_team)
        away_player, away_top6, away_continuity = self._lineup_summary(away_team)
        add_difference("player_rating_diff", home_player, away_player)
        add_difference("player_top6_diff", home_top6, away_top6)
        add_difference("lineup_continuity_diff", home_continuity, away_continuity)

        def rest_days(state: TeamState) -> float:
            if state.last_date is None:
                return 7.0
            return float(np.clip((start_time - state.last_date).days, 4, 30))

        add_difference("rest_days_diff", rest_days(home), rest_days(away))
        home_travel = _travel_distance(home_team, venue)
        away_travel = _travel_distance(away_team, venue)
        row["travel_advantage_km"] = away_travel - home_travel
        row["home__travel_advantage_km"] = home_travel
        row["away__travel_advantage_km"] = away_travel

        home_familiarity = math.log1p(self.venue_games[(home_team, venue)])
        away_familiarity = math.log1p(self.venue_games[(away_team, venue)])
        add_difference("venue_familiarity_diff", home_familiarity, away_familiarity)
        venue_region = VENUE_REGION.get(venue)
        home_state = int(venue_region is not None and TEAM_REGION.get(home_team) == venue_region)
        away_state = int(venue_region is not None and TEAM_REGION.get(away_team) == venue_region)
        row["home_state_advantage"] = float(home_state - away_state)
        row["home__home_state_advantage"] = float(home_state)
        row["away__home_state_advantage"] = float(away_state)
        row["finals_match"] = float(bool(fixture.get("is_final")))
        row["season_progress"] = min(float(fixture.get("round_number") or 0) / 24.0, 1.25)
        row["covid_2020"] = float(season == 2020)
        row["prior_games_min"] = float(min(home.games, away.games))
        return row

    @staticmethod
    def _team_observation(match: dict[str, Any], side: str) -> dict[str, float | None]:
        opponent = "away" if side == "home" else "home"
        stats = match[f"{side}_stats"]
        opponent_stats = match[f"{opponent}_stats"]
        score_for = _number(match[f"{side}_score"])
        score_against = _number(match[f"{opponent}_score"])
        xscore_for = _number(stats.get("xScore"))
        xscore_against = _number(opponent_stats.get("xScore"))
        return {
            "margin": None if score_for is None or score_against is None else score_for - score_against,
            "score_for": score_for,
            "score_against": score_against,
            "xscore_for": xscore_for,
            "xscore_against": xscore_against,
            "finishing": None if score_for is None or xscore_for is None else score_for - xscore_for,
            "inside50": _number(stats.get("Inside50s")),
            "clearance": _number(stats.get("TotalClearances")),
            "centre_clearance": _number(stats.get("CentreClearances")),
            "contested": _number(stats.get("ContestedPossessions")),
            "ground_ball": _number(stats.get("GroundBallGets")),
            "post_clearance": _number(stats.get("PostClearanceContestedPossessions")),
            "disposal_efficiency": _number(stats.get("DisposalEfficiency")),
            "metres_gained": _number(stats.get("MetresGained")),
            "pressure": _number(stats.get("PressureActs")),
            "turnover": _number(stats.get("Turnovers")),
            "intercept": _number(stats.get("Intercepts")),
            "contested_marks": _number(stats.get("ContestedMarks")),
            "shots": _number(stats.get("ShotsAtGoal")),
            "team_rating": _number(stats.get("RatingPoints")),
            "experience": _number(stats.get("Experience")),
        }

    @staticmethod
    def _ewm_update(values: dict[str, float], observations: dict[str, float | None], alpha: float) -> None:
        for key, observation in observations.items():
            if observation is None:
                continue
            previous = values.get(key)
            values[key] = observation if previous is None else alpha * observation + (1 - alpha) * previous

    def _update_players(self, team: str, players: Iterable[dict[str, Any]]) -> list[str]:
        lineup: list[str] = []
        for player in players:
            key = _player_key(player)
            lineup.append(key)
            actual_rating = _number(player.get("RatingPoints"))
            if actual_rating is None:
                continue
            state = self.players[(team, key)]
            prior_weight = min(state.games, 8)
            shrink = (actual_rating + prior_weight * state.rating) / (prior_weight + 1)
            state.rating = 0.30 * shrink + 0.70 * state.rating if state.games else shrink
            state.games += 1
        return lineup

    def update_from_match(self, match: dict[str, Any]) -> None:
        match_id = str(match.get("match_id"))
        if match_id in self.processed_match_ids:
            return
        home_team = canonical_team(match["home_team"])
        away_team = canonical_team(match["away_team"])
        home = self.teams[home_team]
        away = self.teams[away_team]
        home_score = float(match["home_score"])
        away_score = float(match["away_score"])
        margin = home_score - away_score

        expected_home = 1.0 / (1.0 + 10 ** (-((home.elo + 35.0) - away.elo) / 400.0))
        actual_home = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        margin_multiplier = math.log(abs(margin) + 1.0) * 2.2 / (
            abs(home.elo - away.elo) * 0.001 + 2.2
        )
        change = 20.0 * margin_multiplier * (actual_home - expected_home)
        home.elo += change
        away.elo -= change

        for side, team, state in (
            ("home", home_team, home),
            ("away", away_team, away),
        ):
            observations = self._team_observation(match, side)
            self._ewm_update(state.short, observations, alpha=0.30)
            self._ewm_update(state.long, observations, alpha=0.12)
            players = match[f"{side}_players"]
            new_lineup = self._update_players(team, players)
            if state.last_lineup and new_lineup:
                state.continuity = 100.0 * len(set(state.last_lineup) & set(new_lineup)) / len(new_lineup)
            state.last_lineup = new_lineup or state.last_lineup
            state.games += 1
            state.last_date = match["start_time"]
            self.venue_games[(team, str(match.get("venue") or "Unknown venue"))] += 1
        self.processed_match_ids.add(match_id)


def build_feature_frame(matches: list[dict[str, Any]]) -> tuple[pd.DataFrame, FeatureEngine]:
    engine = FeatureEngine()
    rows: list[dict[str, Any]] = []
    batches: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        key = (int(match["season"]), int(match["round_number"]), str(match["round_name"]))
        batches[key].append(match)

    ordered_batches = sorted(
        batches.values(),
        key=lambda batch: min(match["start_time"] for match in batch),
    )
    for batch in ordered_batches:
        engine.begin_season(int(batch[0]["season"]))
        for match in sorted(batch, key=lambda item: item["start_time"]):
            feature_values = engine.features_for_fixture(match)
            rows.append(
                {
                    "match_id": match["match_id"],
                    "source_game_id": match.get("source_game_id"),
                    "season": int(match["season"]),
                    "round_number": int(match["round_number"]),
                    "round_name": match["round_name"],
                    "start_time": match["start_time"],
                    "venue": match["venue"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "home_score": float(match["home_score"]),
                    "away_score": float(match["away_score"]),
                    "home_margin": float(match["home_score"] - match["away_score"]),
                    **feature_values,
                }
            )
        for match in batch:
            engine.update_from_match(match)

    frame = pd.DataFrame(rows).sort_values(["start_time", "match_id"]).reset_index(drop=True)
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    for column in missing:
        frame[column] = np.nan
    return frame, engine


def update_engine_with_matches(engine: FeatureEngine, matches: list[dict[str, Any]]) -> int:
    pending = [
        match
        for match in matches
        if str(match.get("match_id")) not in engine.processed_match_ids
    ]
    batches: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for match in pending:
        key = (int(match["season"]), int(match["round_number"]), str(match["round_name"]))
        batches[key].append(match)
    ordered = sorted(
        batches.values(),
        key=lambda batch: min(match["start_time"] for match in batch),
    )
    for batch in ordered:
        engine.begin_season(int(batch[0]["season"]))
        for match in batch:
            engine.update_from_match(match)
    return len(pending)


def raw_same_match_correlations(matches: list[dict[str, Any]], excluded_season: int) -> list[dict[str, Any]]:
    labels = {
        "xScore": "Expected score",
        "Inside50s": "Inside 50s",
        "TotalClearances": "Clearances",
        "CentreClearances": "Centre clearances",
        "ContestedPossessions": "Contested possessions",
        "DisposalEfficiency": "Disposal efficiency",
        "MetresGained": "Metres gained",
        "PressureActs": "Pressure acts",
        "Turnovers": "Turnovers",
        "Intercepts": "Intercept possessions",
        "ContestedMarks": "Contested marks",
        "ShotsAtGoal": "Scoring shots",
        "RatingPoints": "Team player rating",
    }
    output: list[dict[str, Any]] = []
    margins: list[float] = []
    differences: dict[str, list[float | None]] = {key: [] for key in labels}
    for match in matches:
        if int(match["season"]) == excluded_season:
            continue
        margins.append(float(match["home_score"] - match["away_score"]))
        for key in labels:
            home_value = _number(match["home_stats"].get(key))
            away_value = _number(match["away_stats"].get(key))
            differences[key].append(
                None if home_value is None or away_value is None else home_value - away_value
            )
    for key, values in differences.items():
        paired = [(value, margin) for value, margin in zip(values, margins) if value is not None]
        if len(paired) < 80:
            continue
        x = pd.Series([item[0] for item in paired], dtype=float)
        y = pd.Series([item[1] for item in paired], dtype=float)
        output.append(
            {
                "feature": key,
                "label": labels[key],
                "spearman_correlation": round(float(x.corr(y, method="spearman")), 3),
                "matches": len(paired),
            }
        )
    return sorted(output, key=lambda item: abs(item["spearman_correlation"]), reverse=True)
