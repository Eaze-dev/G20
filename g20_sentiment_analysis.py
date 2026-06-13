"""
=============================================================================
RESEARCH: Assessing Business Opportunity Perceptions During South Africa's
          G20 Presidency Using Social Media Sentiment Analysis & ML
=============================================================================
CRISP-DM Framework | Apify Twitter/X Data Version
=============================================================================
"""

import os, re, json, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    # !! UPDATE THIS PATH to wherever you saved the Apify CSV !!
    "apify_csv":       "data/raw_apify.csv",
    "clean_data_path": "data/cleaned_tweets.csv",
    "results_path":    "results/",
    "figures_path":    "figures/",
    "test_size":       0.2,
    "random_state":    42,
    "cv_folds":        5,
}

for folder in ["data", "results", "figures"]:
    Path(folder).mkdir(exist_ok=True)


# =============================================================================
# PHASE 2 – DATA UNDERSTANDING: Load Apify CSV
# =============================================================================

def load_apify_data(csv_path: str) -> pd.DataFrame:
    """
    Load and standardise the Apify Twitter/X scraper CSV.
    Apify exports a flat CSV with columns like:
      full_text, created_at, favorite_count, retweet_count,
      reply_count, quote_count, author/screen_name,
      author/followers_count, author/location, lang, id
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: LOADING APIFY DATA")
    logger.info("=" * 60)

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Raw shape: {df.shape}")

    # Map Apify columns → our standard names
    rename_map = {
        "full_text":              "text",
        "id":                     "tweet_id",
        "created_at":             "created_at",
        "favorite_count":         "like_count",
        "retweet_count":          "retweet_count",
        "reply_count":            "reply_count",
        "quote_count":            "quote_count",
        "author/screen_name":     "username",
        "author/followers_count": "follower_count",
        "author/location":        "location",
        "lang":                   "lang",
    }

    # Keep only columns that exist
    cols_to_keep = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)

    # Detect retweets
    df["is_retweet"] = df["text"].str.startswith("RT @", na=False)

    # Fill missing engagement columns with 0
    for col in ["like_count", "retweet_count", "reply_count", "quote_count", "follower_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    logger.info(f"Loaded {len(df)} tweets")
    logger.info(f"Retweets: {df['is_retweet'].sum()} ({df['is_retweet'].mean():.1%})")
    logger.info(f"Languages:\n{df['lang'].value_counts().head(5).to_string()}")
    logger.info(f"Date range: {df['created_at'].iloc[0]} → {df['created_at'].iloc[-1]}")
    logger.info(f"Top locations:\n{df['location'].value_counts().head(8).to_string()}")

    # Save raw normalised copy
    df.to_csv(CONFIG["apify_csv"], index=False)
    return df


# =============================================================================
# PHASE 3 – DATA PREPARATION
# =============================================================================

def download_nltk_resources():
    for r in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
              "punkt_tab", "omw-1.4"]:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"RT\s+", "", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize_lemmatize(text: str, stop_words: set) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t.isalpha() and t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: DATA PREPARATION")
    logger.info("=" * 60)

    download_nltk_resources()
    stop_words = set(stopwords.words("english"))
    stop_words.update({"south", "africa", "african", "g20", "sa", "amp", "gt", "lt"})

    df = df.copy()

    # Parse dates
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df.dropna(subset=["created_at", "text"], inplace=True)

    # Remove retweets
    before = len(df)
    df = df[~df["is_retweet"]].copy()
    logger.info(f"Removed {before - len(df)} retweets. Remaining: {len(df)}")

    # Keep English only
    if "lang" in df.columns:
        df = df[df["lang"] == "en"].copy()
        logger.info(f"English-only tweets: {len(df)}")

    # Drop very short tweets
    df = df[df["text"].str.len() > 20].copy()

    # Clean & process text
    df["text_clean"]     = df["text"].apply(clean_text)
    df["text_processed"] = df["text_clean"].apply(
        lambda t: tokenize_lemmatize(t, stop_words))

    # Time features
    df["year"]    = df["created_at"].dt.year
    df["month"]   = df["created_at"].dt.month
    df["weekday"] = df["created_at"].dt.day_name()
    df["hour"]    = df["created_at"].dt.hour

    # Engagement score
    df["engagement"] = (df["like_count"] + df["retweet_count"] * 2 +
                        df["reply_count"] + df["quote_count"])

    logger.info(f"Final clean dataset: {df.shape}")
    df.to_csv(CONFIG["clean_data_path"], index=False)
    return df


# =============================================================================
# PHASE 4 – MODELING
# =============================================================================

def apply_vader(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n── VADER Sentiment ─────────────────────────────────────")
    vader = SentimentIntensityAnalyzer()

    def score(text):
        s = vader.polarity_scores(str(text))
        if   s["compound"] >= 0.05:  return "positive", s["compound"]
        elif s["compound"] <= -0.05: return "negative", s["compound"]
        else:                        return "neutral",   s["compound"]

    df[["vader_sentiment", "vader_compound"]] = df["text_clean"].apply(
        lambda t: pd.Series(score(t)))
    logger.info(f"VADER:\n{df['vader_sentiment'].value_counts().to_string()}")
    return df


def apply_textblob(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n── TextBlob Sentiment ──────────────────────────────────")

    def score(text):
        a = TextBlob(str(text))
        pol = a.sentiment.polarity
        if   pol > 0.05:  return "positive", pol, a.sentiment.subjectivity
        elif pol < -0.05: return "negative", pol, a.sentiment.subjectivity
        else:             return "neutral",   pol, a.sentiment.subjectivity

    df[["textblob_sentiment", "textblob_polarity", "textblob_subjectivity"]] = \
        df["text_clean"].apply(lambda t: pd.Series(score(t)))
    logger.info(f"TextBlob:\n{df['textblob_sentiment'].value_counts().to_string()}")
    return df


def ensemble_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    def combine(row):
        if row["vader_sentiment"] == row["textblob_sentiment"]:
            return row["vader_sentiment"]
        return row["vader_sentiment"]   # VADER wins tie

    df["sentiment_label"] = df.apply(combine, axis=1)
    logger.info(f"\nEnsemble:\n{df['sentiment_label'].value_counts().to_string()}")
    return df


def build_ml_models(df: pd.DataFrame) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: ML MODELING")
    logger.info("=" * 60)

    df = df[df["text_processed"].str.len() > 5].copy()
    label_col = "sentiment_label"   # always use ensemble label

    X = df["text_processed"].values
    y = df[label_col].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"], stratify=y_enc)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (Linear)":        LinearSVC(max_iter=2000, C=1.0),
    }

    results = {}
    cv = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        clf.fit(X_tr, y_train)
        y_pred   = clf.predict(X_te)
        acc      = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average="weighted")
        cv_sc    = cross_val_score(clf, X_tr, y_train, cv=cv, scoring="accuracy")
        report   = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

        results[name] = dict(clf=clf, tfidf=tfidf, le=le,
                             y_test=y_test, y_pred=y_pred,
                             accuracy=acc, f1=f1,
                             cv_mean=cv_sc.mean(), cv_std=cv_sc.std(),
                             report=report)
        logger.info(f"  {name:<22} Acc={acc:.4f}  F1={f1:.4f}  CV={cv_sc.mean():.4f}±{cv_sc.std():.4f}")

    return results


# =============================================================================
# PHASE 5 – EVALUATION
# =============================================================================

def evaluate_and_compare(results: dict) -> str:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 5: MODEL COMPARISON")
    logger.info("=" * 60)

    rows = [{"Model": n,
             "Accuracy": round(r["accuracy"], 4),
             "F1 (weighted)": round(r["f1"], 4),
             "CV Mean": round(r["cv_mean"], 4),
             "CV Std": round(r["cv_std"], 4)}
            for n, r in results.items()]

    cdf = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False)
    logger.info("\n" + cdf.to_string(index=False))
    cdf.to_csv(f"{CONFIG['results_path']}model_comparison.csv", index=False)

    best = cdf.iloc[0]["Model"]
    logger.info(f"\n✅ Best model: {best}")
    return best


# =============================================================================
# PHASE 6 – DEPLOYMENT: VISUALISATIONS & REPORT
# =============================================================================

def plot_sentiment_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"}

    for ax, col, title in zip(axes,
                               ["vader_sentiment", "textblob_sentiment"],
                               ["VADER", "TextBlob"]):
        vc = df[col].value_counts()
        bars = ax.bar(vc.index, vc.values,
                      color=[colors.get(l, "grey") for l in vc.index],
                      edgecolor="white")
        ax.set_title(f"{title} Sentiment Distribution", fontweight="bold")
        ax.set_xlabel("Sentiment"); ax.set_ylabel("Tweets")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    str(b.get_height()), ha="center", fontsize=10)

    plt.suptitle("SA G20 Presidency – Business Perception Sentiment",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_path']}01_sentiment_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: 01_sentiment_distribution.png")


def plot_sentiment_over_time(df):
    df2 = df.copy()
    df2["month_year"] = df2["created_at"].dt.to_period("M").astype(str)
    pivot = df2.groupby(["month_year", "sentiment_label"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"}
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", label=col.capitalize(),
                color=colors.get(col, "grey"), linewidth=2)
    ax.set_title("Sentiment Trend Over Time – SA G20 Presidency", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Tweets")
    ax.legend(); plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_path']}02_sentiment_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: 02_sentiment_over_time.png")


def plot_wordclouds(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, sent, cmap in zip(axes, ["positive", "negative"], ["Greens", "Reds"]):
        text = " ".join(df[df["sentiment_label"] == sent]["text_processed"].dropna())
        if not text.strip():
            text = "no data"
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap=cmap, max_words=100).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{sent.capitalize()} Keywords", fontweight="bold")
    plt.suptitle("Key Terms in Business Perception Tweets", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_path']}03_wordclouds.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: 03_wordclouds.png")


def plot_confusion_matrices(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1: axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(r["y_test"], r["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=r["le"].classes_).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAcc={r['accuracy']:.3f}", fontsize=9)
    plt.suptitle("Confusion Matrices – All Models", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_path']}04_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: 04_confusion_matrices.png")


def plot_model_comparison(results):
    names  = list(results.keys())
    acc    = [r["accuracy"] for r in results.values()]
    f1     = [r["f1"]       for r in results.values()]
    cv     = [r["cv_mean"]  for r in results.values()]
    cv_err = [r["cv_std"]   for r in results.values()]
    x      = np.arange(len(names))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, acc, w, label="Test Accuracy",  color="#3498db")
    ax.bar(x,     f1,  w, label="F1 (weighted)",  color="#2ecc71")
    ax.bar(x + w, cv,  w, label="CV Accuracy",    color="#9b59b6",
           yerr=cv_err, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("ML Model Performance Comparison", fontweight="bold")
    ax.legend(); ax.axhline(0.7, color="red", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_path']}05_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: 05_model_comparison.png")


def plot_engagement_by_sentiment(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"}
    order = [s for s in ["positive", "neutral", "negative"] if s in df["sentiment_label"].values]
    cap = df["engagement"].quantile(0.95)
    data_list = [df[df["sentiment_label"] == s]["engagement"].clip(upper=cap) for s in order]
    bp = ax.boxplot(data_list, labels=[s.capitalize() for s in order],
                    patch_artist=True, medianprops=dict(color="white", linewidth=2))
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(colors.get(s, "grey")); patch.set_alpha(0.7)
    ax.set_title("Tweet Engagement by Sentiment", fontweight="bold")
    ax.set_ylabel("Engagement (likes + 2×RT + replies + quotes)")
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_path']}06_engagement_by_sentiment.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: 06_engagement_by_sentiment.png")


def plot_top_locations(df):
    loc = df[df["location"].notna() & (df["location"] != "")]["location"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    loc.plot(kind="barh", ax=ax, color="#3498db")
    ax.set_title("Top 10 Tweet Locations", fontweight="bold")
    ax.set_xlabel("Number of Tweets")
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_path']}07_top_locations.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: 07_top_locations.png")


def save_report(df, results, best):
    logger.info(f"\n── Best Model Report ({best}) ──")
    logger.info(results[best]["report"])

    df.to_csv(f"{CONFIG['results_path']}final_annotated_tweets.csv", index=False)

    total = len(df)
    summary = {
        "total_tweets":        total,
        "positive_pct":        round(len(df[df["sentiment_label"] == "positive"]) / total * 100, 1),
        "negative_pct":        round(len(df[df["sentiment_label"] == "negative"]) / total * 100, 1),
        "neutral_pct":         round(len(df[df["sentiment_label"] == "neutral"])  / total * 100, 1),
        "best_ml_model":       best,
        "best_model_accuracy": round(results[best]["accuracy"], 4),
        "best_model_f1":       round(results[best]["f1"], 4),
        "avg_engagement":      round(df["engagement"].mean(), 1),
        "top_location":        df["location"].value_counts().index[0] if "location" in df.columns else "N/A",
        "date_first_tweet":    str(df["created_at"].min()),
        "date_last_tweet":     str(df["created_at"].max()),
    }

    with open(f"{CONFIG['results_path']}summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("RESEARCH SUMMARY")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k:<30} {v}")
    logger.info("=" * 60)
    logger.info("✅ All outputs saved to ./data/, ./results/, ./figures/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "═" * 60)
    print("  SA G20 PRESIDENCY – BUSINESS PERCEPTION SENTIMENT ANALYSIS")
    print("  CRISP-DM Framework | Real Apify Data")
    print("═" * 60 + "\n")

    # Phase 2 – Load
    df_raw = load_apify_data(CONFIG["apify_csv"])

    # Phase 3 – Prepare
    df = prepare_data(df_raw)

    # Phase 4 – Model
    df = apply_vader(df)
    df = apply_textblob(df)
    df = ensemble_sentiment(df)
    ml_results = build_ml_models(df)

    # Phase 5 – Evaluate
    best = evaluate_and_compare(ml_results)

    # Phase 6 – Deploy
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 6: GENERATING VISUALISATIONS")
    logger.info("=" * 60)
    plot_sentiment_distribution(df)
    plot_sentiment_over_time(df)
    plot_wordclouds(df)
    plot_confusion_matrices(ml_results)
    plot_model_comparison(ml_results)
    plot_engagement_by_sentiment(df)
    plot_top_locations(df)
    save_report(df, ml_results, best)


if __name__ == "__main__":
    main()
