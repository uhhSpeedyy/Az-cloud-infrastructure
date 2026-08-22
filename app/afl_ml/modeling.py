from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, FEATURE_INFO, FeatureEngine


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    result = np.array(
        [0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in flat],
        dtype=float,
    )
    return result.reshape(np.asarray(values).shape)


def _probabilities(mean_margin: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_margin = np.asarray(mean_margin, dtype=float)
    away = _normal_cdf((-0.5 - mean_margin) / sigma)
    home = 1.0 - _normal_cdf((0.5 - mean_margin) / sigma)
    draw = np.clip(1.0 - away - home, 0.0, 1.0)
    return home, away, draw


def _build_ridge() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=24.0)),
        ]
    )


def _build_tree() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.045,
                    max_iter=260,
                    max_leaf_nodes=15,
                    min_samples_leaf=24,
                    l2_regularization=8.0,
                    random_state=42,
                ),
            ),
        ]
    )


def _fit_pair(x: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, Pipeline]:
    ridge = _build_ridge()
    tree = _build_tree()
    ridge.fit(x, y)
    tree.fit(x, y)
    return ridge, tree


def _blend_predict(
    ridge: Pipeline,
    tree: Pipeline,
    x: pd.DataFrame,
    ridge_weight: float,
) -> np.ndarray:
    return ridge_weight * ridge.predict(x) + (1.0 - ridge_weight) * tree.predict(x)


@dataclass
class ModelBundle:
    ridge: Pipeline
    tree: Pipeline
    ridge_weight: float
    calibration_intercept: float
    calibration_slope: float
    residual_sigma: float
    feature_columns: list[str]
    feature_importance: dict[str, float]
    feature_correlations: dict[str, float]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    feature_engine: FeatureEngine
    model_version: str
    trained_at: str
    holdout_season: int
    metrics: dict[str, Any]

    def predict_margin(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.reindex(columns=self.feature_columns)
        raw = _blend_predict(self.ridge, self.tree, x, self.ridge_weight)
        return self.calibration_intercept + self.calibration_slope * raw

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        margin = self.predict_margin(frame)
        home, away, draw = _probabilities(margin, self.residual_sigma)
        return {
            "margin": margin,
            "home_probability": home,
            "away_probability": away,
            "draw_probability": draw,
            "interval_low": margin - 1.2816 * self.residual_sigma,
            "interval_high": margin + 1.2816 * self.residual_sigma,
        }

    def explain_row(self, row: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
        frame = pd.DataFrame([row]).reindex(columns=self.feature_columns)
        transformed = self.ridge.named_steps["imputer"].transform(frame)
        transformed = self.ridge.named_steps["scaler"].transform(transformed)
        coefficients = self.ridge.named_steps["model"].coef_
        contributions = (
            self.ridge_weight
            * self.calibration_slope
            * transformed[0, : len(self.feature_columns)]
            * coefficients[: len(self.feature_columns)]
        )
        predicted_margin = float(self.predict_margin(frame)[0])
        winner_sign = 1.0 if predicted_margin >= 0 else -1.0
        ranked: list[tuple[float, int]] = []
        for index, feature in enumerate(self.feature_columns):
            value = row.get(feature)
            if pd.isna(value):
                continue
            support = float(contributions[index]) * winner_sign
            if feature in {"finals_match", "season_progress", "covid_2020", "prior_games_min"}:
                continue
            ranked.append((support, index))
        supporting = [item for item in ranked if item[0] > 0]
        chosen = sorted(supporting or ranked, reverse=True)[:limit]

        explanations: list[dict[str, Any]] = []
        for _, index in chosen:
            feature = self.feature_columns[index]
            info = FEATURE_INFO[feature]
            contribution = float(contributions[index])
            home_value = row.get(f"home__{feature}")
            away_value = row.get(f"away__{feature}")
            if pd.isna(home_value) if home_value is not None else False:
                home_value = None
            if pd.isna(away_value) if away_value is not None else False:
                away_value = None
            advantage_team = row["home_team"] if contribution >= 0 else row["away_team"]
            explanations.append(
                {
                    "feature": feature,
                    "label": info["label"],
                    "unit": info["unit"],
                    "description": info["description"],
                    "home_value": None if home_value is None else round(float(home_value), 1),
                    "away_value": None if away_value is None else round(float(away_value), 1),
                    "difference": round(float(row[feature]), 1),
                    "advantage_team": advantage_team,
                    "linear_contribution_points": round(contribution, 1),
                    "lower_is_better": bool(info.get("lower_is_better", False)),
                }
            )
        return explanations


def _calibrate(oof_prediction: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(oof_prediction, actual, deg=1)
    slope = float(np.clip(slope, 0.55, 1.45))
    intercept = float(np.clip(intercept, -10.0, 10.0))
    calibrated = intercept + slope * oof_prediction
    outcomes = (actual > 0).astype(float)
    non_draw = actual != 0
    best_sigma = 28.0
    best_brier = float("inf")
    for sigma in np.linspace(15.0, 48.0, 133):
        home_probability, _, _ = _probabilities(calibrated, float(sigma))
        brier = float(np.mean((home_probability[non_draw] - outcomes[non_draw]) ** 2))
        if brier < best_brier:
            best_brier = brier
            best_sigma = float(sigma)
    return intercept, slope, best_sigma


def _metrics(actual: np.ndarray, predicted: np.ndarray, sigma: float) -> dict[str, float | int]:
    error = predicted - actual
    home_probability, _, _ = _probabilities(predicted, sigma)
    non_draw = actual != 0
    outcomes = (actual > 0).astype(float)
    accuracy = np.mean((home_probability[non_draw] >= 0.5) == outcomes[non_draw])
    clipped = np.clip(home_probability[non_draw], 1e-6, 1 - 1e-6)
    log_loss = -np.mean(
        outcomes[non_draw] * np.log(clipped)
        + (1 - outcomes[non_draw]) * np.log(1 - clipped)
    )
    return {
        "matches": int(len(actual)),
        "tip_accuracy": round(float(accuracy), 4),
        "brier_score": round(float(np.mean((home_probability[non_draw] - outcomes[non_draw]) ** 2)), 4),
        "log_loss": round(float(log_loss), 4),
        "margin_mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "margin_rmse": round(float(math.sqrt(mean_squared_error(actual, predicted))), 2),
        "margin_bias": round(float(np.mean(error)), 2),
        "within_12_points": round(float(np.mean(np.abs(error) <= 12)), 4),
        "within_24_points": round(float(np.mean(np.abs(error) <= 24)), 4),
        "interval_80_coverage": round(float(np.mean(np.abs(error) <= 1.2816 * sigma)), 4),
    }


def _permutation_importance(
    bundle: ModelBundle,
    holdout: pd.DataFrame,
    repeats: int = 8,
) -> dict[str, float]:
    x = holdout[bundle.feature_columns].copy()
    actual = holdout["home_margin"].to_numpy(dtype=float)
    baseline = mean_absolute_error(actual, bundle.predict_margin(x))
    rng = np.random.default_rng(2022)
    importance: dict[str, float] = {}
    for column in bundle.feature_columns:
        values = x[column].to_numpy(copy=True)
        scores: list[float] = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[column] = rng.permutation(values)
            score = mean_absolute_error(actual, bundle.predict_margin(shuffled))
            scores.append(float(score - baseline))
        importance[column] = round(float(np.mean(scores)), 4)
    return importance


def train_model(
    frame: pd.DataFrame,
    feature_engine: FeatureEngine,
    holdout_season: int = 2022,
) -> tuple[ModelBundle, dict[str, Any]]:
    usable = frame.loc[frame["prior_games_min"] >= 3].copy()
    development = usable.loc[usable["season"] < holdout_season].copy()
    holdout = usable.loc[usable["season"] == holdout_season].copy()
    if development.empty or holdout.empty:
        raise ValueError("Training and holdout seasons must both contain matches")

    validation_seasons = [
        season
        for season in sorted(development["season"].unique())
        if season >= 2017
    ]
    oof_actual: list[float] = []
    oof_ridge: list[float] = []
    oof_tree: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for season in validation_seasons:
        train = development.loc[development["season"] < season]
        validation = development.loc[development["season"] == season]
        if len(train) < 400 or validation.empty:
            continue
        ridge, tree = _fit_pair(train[FEATURE_COLUMNS], train["home_margin"])
        ridge_prediction = ridge.predict(validation[FEATURE_COLUMNS])
        tree_prediction = tree.predict(validation[FEATURE_COLUMNS])
        oof_actual.extend(validation["home_margin"].astype(float))
        oof_ridge.extend(ridge_prediction)
        oof_tree.extend(tree_prediction)
        fold_rows.append(
            {
                "season": int(season),
                "train_matches": int(len(train)),
                "validation_matches": int(len(validation)),
            }
        )

    oof_actual_array = np.asarray(oof_actual, dtype=float)
    oof_ridge_array = np.asarray(oof_ridge, dtype=float)
    oof_tree_array = np.asarray(oof_tree, dtype=float)
    if not len(oof_actual_array):
        raise ValueError("No chronological validation predictions were produced")

    blend_candidates = (0.25, 0.40, 0.55, 0.70, 0.85)
    ridge_weight = min(
        blend_candidates,
        key=lambda weight: mean_absolute_error(
            oof_actual_array,
            weight * oof_ridge_array + (1 - weight) * oof_tree_array,
        ),
    )
    oof_blend = ridge_weight * oof_ridge_array + (1 - ridge_weight) * oof_tree_array
    intercept, slope, sigma = _calibrate(oof_blend, oof_actual_array)
    oof_calibrated = intercept + slope * oof_blend

    development_ridge, development_tree = _fit_pair(
        development[FEATURE_COLUMNS],
        development["home_margin"],
    )
    holdout_raw = _blend_predict(
        development_ridge,
        development_tree,
        holdout[FEATURE_COLUMNS],
        ridge_weight,
    )
    holdout_prediction = intercept + slope * holdout_raw
    holdout_metrics = _metrics(
        holdout["home_margin"].to_numpy(dtype=float),
        holdout_prediction,
        sigma,
    )

    production_training = usable.loc[usable["season"] != holdout_season].copy()
    final_ridge, final_tree = _fit_pair(
        production_training[FEATURE_COLUMNS],
        production_training["home_margin"],
    )
    trained_at = datetime.now(timezone.utc).isoformat()
    model_version = datetime.now(timezone.utc).strftime("afl-%Y%m%d-%H%M%S")
    correlations: dict[str, float] = {}
    for feature in FEATURE_COLUMNS:
        value = production_training[[feature, "home_margin"]].corr(method="spearman").iloc[0, 1]
        correlations[feature] = 0.0 if pd.isna(value) else float(value)
    bundle = ModelBundle(
        ridge=final_ridge,
        tree=final_tree,
        ridge_weight=float(ridge_weight),
        calibration_intercept=intercept,
        calibration_slope=slope,
        residual_sigma=sigma,
        feature_columns=list(FEATURE_COLUMNS),
        feature_importance={},
        feature_correlations=correlations,
        feature_means={
            feature: (
                0.0
                if pd.isna(production_training[feature].mean(skipna=True))
                else float(production_training[feature].mean(skipna=True))
            )
            for feature in FEATURE_COLUMNS
        },
        feature_stds={
            feature: (
                1.0
                if pd.isna(production_training[feature].std(skipna=True))
                or production_training[feature].std(skipna=True) == 0
                else float(production_training[feature].std(skipna=True))
            )
            for feature in FEATURE_COLUMNS
        },
        feature_engine=feature_engine,
        model_version=model_version,
        trained_at=trained_at,
        holdout_season=holdout_season,
        metrics=holdout_metrics,
    )
    importance_bundle = ModelBundle(
        ridge=development_ridge,
        tree=development_tree,
        ridge_weight=bundle.ridge_weight,
        calibration_intercept=bundle.calibration_intercept,
        calibration_slope=bundle.calibration_slope,
        residual_sigma=bundle.residual_sigma,
        feature_columns=bundle.feature_columns,
        feature_importance={},
        feature_correlations=bundle.feature_correlations,
        feature_means=bundle.feature_means,
        feature_stds=bundle.feature_stds,
        feature_engine=feature_engine,
        model_version=bundle.model_version,
        trained_at=bundle.trained_at,
        holdout_season=bundle.holdout_season,
        metrics=bundle.metrics,
    )
    bundle.feature_importance = _permutation_importance(importance_bundle, holdout)

    oof_metrics = _metrics(oof_actual_array, oof_calibrated, sigma)
    top_features = sorted(
        (
            {
                "feature": feature,
                "label": FEATURE_INFO[feature]["label"],
                "mae_increase_when_shuffled": importance,
                "lagged_spearman_correlation": round(correlations.get(feature, 0.0), 3),
            }
            for feature, importance in bundle.feature_importance.items()
        ),
        key=lambda item: item["mae_increase_when_shuffled"],
        reverse=True,
    )
    coverage = {
        str(int(season)): round(
            float(frame.loc[frame["season"] == season, FEATURE_COLUMNS].notna().mean().mean()),
            3,
        )
        for season in sorted(frame["season"].unique())
    }
    report = {
        "model_version": model_version,
        "trained_at": trained_at,
        "holdout_season": holdout_season,
        "holdout_policy": (
            "Estimator parameters and tuning were frozen using 2012-2021 before the "
            "one-pass 2022 test. Production fitting excludes 2022 rows."
        ),
        "training_seasons": sorted(
            int(season) for season in production_training["season"].unique()
        ),
        "training_matches": int(len(production_training)),
        "holdout_metrics": holdout_metrics,
        "chronological_validation_metrics": oof_metrics,
        "validation_folds": fold_rows,
        "ridge_blend_weight": float(ridge_weight),
        "tree_blend_weight": float(1 - ridge_weight),
        "residual_sigma_points": round(float(sigma), 2),
        "feature_coverage_by_season": coverage,
        "top_predictive_features": top_features,
    }
    return bundle, report
