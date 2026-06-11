"""
classifier/train.py
───────────────────
Fetch training data from PostgreSQL, train a char-ngram TF-IDF +
LogisticRegression pipeline, and persist versioned artefacts to disk.

Versioning:
  - Models are saved under  classifier/models/model_v<N>.pkl
  - The active symlink      classifier/models/model_latest.pkl  always points
    to the newest version so predictor.py never needs to know the version number.
  - Only the last MAX_KEPT_VERSIONS versions are retained (Docker image budget).
  - Every version is recorded in the insight_classifier_models DB table for auditing
    and rollback.

Run manually:
    python -m app.classifier.train

Or call  POST /classifier/retrain  for an in-process hot-swap.
"""

import logging
import re
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from app.utils.db import get_db

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).parent
MODELS_DIR    = _HERE / "models"
LATEST_LINK   = MODELS_DIR / "model_latest.pkl"

MAX_KEPT_VERSIONS = 3   # keep last N .pkl files inside the container

# ── Noise words that appear in every category — strip them before vectorising ──
_NOISE_WORDS = {
    "the","a","an","is","was","were","be","been","or","and",
    "of","to","in","on","at","for"
}


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Normalise raw alert text / error messages into a clean token string.

    Steps:
      1. Lower-case
      2. Strip URLs and numeric thresholds  (e.g. "100.0 (04/06/26 09:53:00)")
      3. Remove punctuation / special chars  → spaces
      4. Collapse whitespace
      5. Drop noise words
    """
    t = text.lower().strip()

    # Remove URLs
    t = re.sub(r"https?://\S+", " ", t)
    # Remove numeric expressions with units / timestamps
    t = re.sub(r"\d[\d:./()\[\]%,\-]*", " ", t)
    # Remove common error prefixes like "RequestError:" that add no signal
    t = re.sub(r"\w+error\s*:", " ", t)
    # Replace non-alphanumeric chars with space
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Drop noise words
    tokens = [tok for tok in t.split() if tok not in _NOISE_WORDS and len(tok) > 1]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_training_data() -> tuple[list[str], list[str]]:
    """Return (alert_texts, categories) after dedup and normalisation."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # DISTINCT prevents duplicate feedback from biasing the model
            cur.execute(
                """
                SELECT DISTINCT ON (down_message, category) down_message, category
                FROM meyiconnect.insight_alert_categories_training
                ORDER BY down_message, category, id
                """
            )
            rows = cur.fetchall()

    if not rows:
        raise ValueError(
            "insight_alert_categories_training table is empty — seed data first."
        )

    X = [normalize(row["down_message"]) for row in rows]
    y = [row["category"] for row in rows]
    return X, y


def _next_version() -> int:
    """Return the next integer version by querying the DB or existing model files."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT version FROM meyiconnect.insight_classifier_models
                    ORDER BY id DESC LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row and row["version"]:
                    try:
                        return int(row["version"].replace("v", "")) + 1
                    except ValueError:
                        pass
    except Exception as exc:
        logger.warning("Could not fetch latest version from DB: %s", exc)

    # Fallback to local files
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(MODELS_DIR.glob("model_v*.pkl"))
    if not existing:
        return 1
    
    versions = []
    for p in existing:
        try:
            versions.append(int(p.stem.split("_v")[-1]))
        except ValueError:
            pass
            
    if not versions:
        return 1
    return max(versions) + 1


def _prune_old_versions() -> None:
    """Delete oldest versions, keeping only MAX_KEPT_VERSIONS on disk."""
    existing = list(MODELS_DIR.glob("model_v*.pkl"))
    
    def _get_ver(p: Path) -> int:
        try:
            return int(p.stem.split("_v")[-1])
        except ValueError:
            return -1
            
    versioned = sorted(existing, key=_get_ver)
    to_delete = versioned[:-MAX_KEPT_VERSIONS]
    for path in to_delete:
        path.unlink(missing_ok=True)
        logger.info("Pruned old model version: %s", path.name)


def _record_in_db(version: int, sample_count: int) -> None:
    """Write a row to insight_classifier_models for auditing."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO meyiconnect.insight_classifier_models (version, training_samples)
                    VALUES (%s, %s)
                    """,
                    (f"v{version}", sample_count),
                )
    except Exception as exc:
        # Non-fatal — model is already saved; just log and continue
        logger.warning("Could not record model version in DB: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipeline() -> Pipeline:
    """
    Two-branch TF-IDF:
      • char_wb  ngrams (3-5)  — catches partial word matches, handles typos
        and long error strings like "getaddrinfo ENOTFOUND …"
      • word     ngrams (1-2)  — preserves whole-token signal (cpu, rds, pod …)
    Combined with balanced LogisticRegression.
    """
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        max_features=40_000,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    return Pipeline([
        ("features", FeatureUnion([
            ("char", char_tfidf),
            ("word", word_tfidf),
        ])),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            C=5.0,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save() -> Path:
    """
    Full train → version → save cycle.

    Returns the path to the newly written model file.
    """
    logger.info("Classifier training started")

    X, y = _fetch_training_data()
    n_samples   = len(X)
    n_categories = len(set(y))
    logger.info("Loaded %d unique training samples across %d categories",
                n_samples, n_categories)

    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    # ── Versioned save ────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version    = _next_version()
    model_path = MODELS_DIR / f"model_v{version}.pkl"

    joblib.dump(pipeline, model_path)
    logger.info("Model saved → %s", model_path)

    # Update latest symlink (atomic on POSIX; fine inside a single container)
    tmp_link = MODELS_DIR / "_model_latest_tmp.pkl"
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(model_path.name)
    tmp_link.replace(LATEST_LINK)
    logger.info("Latest link → %s", LATEST_LINK)

    _prune_old_versions()
    _record_in_db(version, n_samples)

    logger.info("Classifier training complete (version=v%d)", version)
    return model_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    train_and_save()