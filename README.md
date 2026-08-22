# Sam Speed's models

Two models run in one Flask application:

- an AFL outcome and margin model; and
- a personalised, metadata-based book recommender.

Choose a model at [sam-speed.azurewebsites.net](https://sam-speed.azurewebsites.net/). AFL predictions are at `/afl`; book recommendations are at `/books`.

## AFL model

- Historical window: 2012–2026, with 3,077 completed matches in the current snapshot.
- Untouched test season: 2022. Model selection and calibration use only data available before 2022; the production estimator excludes all 2022 match rows.
- Estimator: a calibrated blend of regularised linear regression and histogram gradient boosting.
- Inputs: opponent-adjusted results strength, rolling attack/defence, xScore, inside 50s, clearances, contested ball, scoring shots, player ratings, recent-lineup strength and continuity, rest, venue familiarity, and travel.
- Leakage protection: every match feature is a lagged or rolling value known before the bounce. Matches in the same round are calculated from the same pre-round state.
- Current 2022 holdout result: 75.2% tip accuracy and 25.32-point margin MAE across 207 matches.

The full reproducible evaluation and feature-importance output is stored in `app/artifacts/model_report.json`. Same-match correlations in that report are descriptive only; they are not used as same-game inputs and do not establish causation.

## Book recommender

The content-based book recommender uses up to ten favourite books and compares:

- normalised themes and detailed Open Library subjects;
- metadata-derived writing-style proxies;
- author;
- broad length band;
- publication era and language; and
- bounded reader-interest evidence, used as a quality prior rather than a substitute for similarity.

It excludes selected books, limits repeated authors and returns one main list plus up to three theme lists. Each recommendation explains why it matched.

Recommendations stay in the favourites' language. Missing language data defaults to English, and English edition titles are used where available.

Open Library does not provide prose-level style data or user-level ratings. Style is therefore an estimate based on metadata.

Search uses the [Open Library Search API](https://openlibrary.org/dev/docs/api/search) with bounded results, timeouts, retries, caching and rate limiting. A local catalogue is used if Open Library is unavailable. Set `OPEN_LIBRARY_CONTACT` to a monitored email to identify API traffic.

## Data

The AFL model uses derived match, team and player data from [Wheelo Ratings](https://www.wheeloratings.com/) and fixtures/results from the [Squiggle API](https://api.squiggle.com.au/). Requests are server-side, cached and low volume.

Before treating the scheduled Wheelo ingestion as a long-term public production feed, obtain written confirmation that this automated derived use is acceptable. AFL Tables and the fitzRoy/TORP datasets are documented fallback and validation sources.

## Local use

From `app/`, create a virtual environment, install `requirements-dev.txt`, then run:

```bash
python scripts/train_model.py
pytest -q
gunicorn --bind 127.0.0.1:8000 --workers 1 --threads 4 app:app
```

Training runs locally and does not require Azure ML or paid compute. Source snapshots are cached under `app/data/raw/` and are intentionally excluded from Git. The trained model, evaluation report, and next-round prediction snapshot under `app/artifacts/` are deployed with the app.

Optional environment settings can be placed in `app/.env`:

```dotenv
DB_SERVER=speedserver.database.windows.net
DB_NAME=DB_one
AFL_DATABASE_ENABLED=true
AFL_HOLDOUT_SEASON=2022
AFL_START_SEASON=2012
AFL_CURRENT_SEASON=2026
SQUIGGLE_CONTACT=your-contact-address
AFL_REFRESH_TOKEN=use-a-long-random-secret
OPEN_LIBRARY_CONTACT=monitored-contact@example.com
```

No SQL password is stored. Local scripts use the signed-in Azure CLI identity; App Service uses its system-assigned managed identity.

## Azure SQL setup

Run the following once while connected to `DB_one` as its Microsoft Entra administrator:

1. `app/sql/001_afl_schema.sql`
2. `app/sql/002_grant_app_identity.sql`

The second script creates the `Sam-Speed` managed-identity user and grants only `SELECT`, `INSERT`, and `UPDATE` on the AFL model tables. It does not grant broad database roles or delete access.

To initialise from the command line after the database variables are configured:

```bash
python scripts/init_database.py
python scripts/grant_app_identity.py
python scripts/train_model.py --persist-db
```

## Deployment and refresh

`main.tf` configures the existing Azure connection point, Python 3.11 runtime, managed identity, VNet integration, health check, and App Service settings. `.github/workflows/deploy.yml` tests the application and loads the committed model artifact under Python 3.11 before deploying.

The book feature uses the same Flask/Gunicorn process and needs no new Terraform resources, database tables, paid AI API, JVM, or native service. Keeping it in Python also preserves the existing Azure build path; scikit-learn/NumPy already provide compiled numerical components where the AFL model needs them.

`.github/workflows/refresh-predictions.yml` refreshes the current-season state and next round every Tuesday. Add the same long random value as `AFL_REFRESH_TOKEN` in both the App Service settings and the GitHub repository Actions secrets. The endpoint is authenticated and accepts only one refresh at a time.

Useful endpoints:

- `/` — model chooser
- `/afl` — AFL prediction website
- `/books` — interactive book recommender
- `/api/books/search?q=...` — bounded favourite-book search
- `/api/books/recommend` — explainable themed recommendations
- `/api/books/model` — book model card and limitations
- `/api/predictions` — current prediction snapshot
- `/api/model` — model card and evaluation
- `/health` — application health
- `/ready` — SQL readiness without exposing connection details

Predictions are probabilistic and intended for analysis and entertainment, not betting advice.
