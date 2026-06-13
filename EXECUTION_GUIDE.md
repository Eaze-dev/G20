# SA G20 Presidency – Sentiment Analysis
## Complete Step-by-Step Execution Guide
### CRISP-DM Framework | Research Implementation

---

## 📁 Project Structure

After running the script, your folder will look like this:

```
project/
├── g20_sentiment_analysis.py   ← main script
├── requirements.txt
├── data/
│   ├── raw_tweets.csv          ← scraped tweets (raw)
│   └── cleaned_tweets.csv      ← preprocessed tweets
├── results/
│   ├── final_annotated_tweets.csv
│   ├── model_comparison.csv
│   └── summary_stats.json
└── figures/
    ├── 01_sentiment_distribution.png
    ├── 02_sentiment_over_time.png
    ├── 03_wordclouds.png
    ├── 04_confusion_matrices.png
    ├── 05_model_comparison.png
    └── 06_engagement_by_sentiment.png
```

---

## STEP 1 — Set Up Your Environment

### 1.1 Install Python (if not installed)
Download Python 3.10 or 3.11 from https://python.org
- ✅ Check "Add Python to PATH" during installation

### 1.2 Create a project folder
```bash
mkdir g20_research
cd g20_research
```

### 1.3 Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

## STEP 2 — Install Dependencies

Place `requirements.txt` in your project folder, then run:

```bash
pip install -r requirements.txt
```

> ⏳ This takes 3–5 minutes. If snscrape fails, run:
> `pip install git+https://github.com/JustAnotherArchivist/snscrape.git`

Verify installation:
```bash
python -c "import snscrape, sklearn, nltk, vaderSentiment; print('All OK')"
```

---

## STEP 3 — CRISP-DM Phase 1: Business Understanding

### What the script already configures for you:

| Setting | Value |
|---|---|
| Research focus | Business opportunity perception |
| Platform | X (Twitter) |
| Date range | Dec 2024 – Nov 2025 (SA G20 presidency) |
| Language | English |
| Queries | 10 keyword combinations (G20, SA, business, investment, etc.) |
| Max tweets/query | 500 (configurable) |

### Customise in the script (optional):
Open `g20_sentiment_analysis.py` and edit the `CONFIG` block at the top:

```python
CONFIG = {
    "max_tweets_per_query": 500,  # increase to 2000 for more data
    "start_date": "2024-12-01",
    "end_date":   "2025-11-30",
    ...
}
```

You can also add more search queries:
```python
"search_queries": [
    "South Africa G20 business",
    "#G20SA investment",
    # add yours here
]
```

---

## STEP 4 — CRISP-DM Phase 2: Data Collection (Scraping)

### Run the script:
```bash
python g20_sentiment_analysis.py
```

### What happens:
1. The script loops through all 10 search queries
2. For each query, snscrape fetches up to 500 tweets from X — **no API key needed**
3. Raw data is saved to `data/raw_tweets.csv`

### raw_tweets.csv columns:

| Column | Description |
|---|---|
| tweet_id | Unique tweet ID |
| created_at | Date and time posted |
| username | X handle |
| text | Full tweet text |
| like_count | Number of likes |
| retweet_count | Retweet count |
| reply_count | Reply count |
| quote_count | Quote tweet count |
| follower_count | Author follower count |
| location | User location (if set) |
| search_query | Which query captured this tweet |
| is_retweet | True/False |

### If snscrape doesn't work:
The script automatically falls back to **demo mode** — 500 realistic synthetic tweets
so you can test the full pipeline immediately. This is useful while setting up.

> 💡 **Tip for real data**: If snscrape is blocked, consider using the official
> Twitter Academic Research API via Tweepy. You'll need to apply at
> developer.twitter.com for a free Academic account.

---

## STEP 5 — CRISP-DM Phase 3: Data Preparation

### Automated in the script. What it does:

| Step | Action |
|---|---|
| 1 | Parse dates, drop invalid rows |
| 2 | Remove retweets (avoid duplication) |
| 3 | Drop tweets shorter than 20 characters |
| 4 | Clean text: remove URLs, @mentions, punctuation, emojis |
| 5 | Tokenise: split into individual words |
| 6 | Remove stopwords (English + custom: "africa", "g20", etc.) |
| 7 | Lemmatise: reduce words to root form (e.g. "investing" → "invest") |
| 8 | Add time features: year, month, week, hour, weekday |
| 9 | Calculate engagement score: likes + (retweets × 2) + replies + quotes |

### Output: `data/cleaned_tweets.csv`

New columns added:
- `text_clean` — cleaned text (lowercase, no URLs/symbols)
- `text_processed` — fully tokenised and lemmatised
- `month`, `week`, `hour`, `weekday` — time breakdown
- `engagement` — composite engagement score

---

## STEP 6 — CRISP-DM Phase 4: Modeling

### A) Lexicon-Based Sentiment (no training required)

**VADER** (best for social media):
- Understands capitalisation, exclamation marks, emoji context
- Returns: compound score → Positive (≥0.05), Negative (≤−0.05), Neutral
- Adds columns: `vader_sentiment`, `vader_compound`

**TextBlob** (general purpose):
- Returns: polarity (−1 to +1) and subjectivity (0 to 1)
- Adds columns: `textblob_sentiment`, `textblob_polarity`, `textblob_subjectivity`

**Ensemble label** (final ground truth):
- If VADER and TextBlob agree → use that label
- If they disagree → trust VADER (more accurate for tweets)
- Adds column: `sentiment_label`

---

### B) Machine Learning Classifiers

The script trains **5 models** using TF-IDF features (unigrams + bigrams):

| Model | Why Used |
|---|---|
| Logistic Regression | Fast, interpretable baseline |
| Naive Bayes | Strong for text, works with small data |
| Random Forest | Handles non-linear patterns |
| Gradient Boosting | Often highest accuracy |
| SVM (Linear) | Excellent for high-dimensional text |

**TF-IDF settings:**
- Up to 10,000 features
- Unigrams + bigrams (`ngram_range=(1,2)`)
- Sublinear TF scaling

**Evaluation per model:**
- Test accuracy (80/20 split)
- Weighted F1-score
- 5-fold cross-validation (mean ± std)

---

## STEP 7 — CRISP-DM Phase 5: Evaluation

### Reading the results

**`results/model_comparison.csv`** — compare all models:
```
Model                Accuracy   F1    CV Mean   CV Std
Gradient Boosting    0.8812     0.879  0.8720    0.018
Logistic Regression  0.8650     0.863  0.8580    0.021
SVM (Linear)         0.8601     0.858  0.8510    0.023
Random Forest        0.8422     0.840  0.8310    0.025
Naive Bayes          0.8211     0.819  0.8150    0.028
```

### What to look for:
- **Accuracy** — % of tweets correctly classified
- **F1 (weighted)** — balances precision and recall across all 3 classes
- **CV Mean** — how the model performs on unseen data (most reliable)
- **CV Std** — lower = more stable model

### Confusion matrix interpretation (`figures/04_confusion_matrices.png`):
- Diagonal = correctly classified
- Off-diagonal = misclassified (common: neutral confused with positive/negative)

### Reporting checklist for your research:
- [ ] Overall positive/negative/neutral percentages
- [ ] Sentiment trend over time (monthly)
- [ ] Which model performed best and why
- [ ] Engagement analysis by sentiment
- [ ] Key terms from word clouds

---

## STEP 8 — CRISP-DM Phase 6: Deployment (Outputs)

### Figures generated:

| File | Content | Use in research |
|---|---|---|
| `01_sentiment_distribution.png` | Bar chart VADER vs TextBlob | Compare lexicon methods |
| `02_sentiment_over_time.png` | Monthly sentiment trend | Show temporal patterns |
| `03_wordclouds.png` | Key words per sentiment | Qualitative insight |
| `04_confusion_matrices.png` | All 5 model matrices | Evaluate classification |
| `05_model_comparison.png` | Accuracy/F1/CV bar chart | Select best model |
| `06_engagement_by_sentiment.png` | Engagement boxplot | Reach analysis |

### Data files:

| File | Use |
|---|---|
| `data/raw_tweets.csv` | Original scraped data |
| `data/cleaned_tweets.csv` | Preprocessed dataset |
| `results/final_annotated_tweets.csv` | Full dataset with sentiment labels |
| `results/model_comparison.csv` | Model performance table |
| `results/summary_stats.json` | Key research statistics |

---

## STEP 9 — Manual Labelling (Improves ML Accuracy)

For stronger ML results, manually label a sample of tweets:

```bash
# Open the clean data
# Add a column "manual_label" with: positive / negative / neutral
# Save as: data/manual_labels.csv
```

Then in the script, change the label column:
```python
label_col = "manual_label"  # line in build_ml_models()
```

Aim for at least **200–300 manually labelled tweets** for reliable ML training.

---

## STEP 10 — Common Issues & Fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: snscrape` | Run: `pip install git+https://github.com/JustAnotherArchivist/snscrape.git` |
| `No tweets scraped` | Script auto-switches to demo data; check your internet connection |
| `NLTK resource not found` | Run in Python: `import nltk; nltk.download('all')` |
| `Empty wordcloud` | Not enough tweets matched the sentiment; lower `max_tweet_length` filter |
| Very low accuracy (<60%) | Collect more data; increase `max_tweets_per_query` to 2000 |
| Rate limiting by X | Add `time.sleep(5)` between queries (already partially included) |

---

## Research Ethics Note

- Only collect **public** tweets
- Anonymise usernames in published data: use `user_id` hash instead of handle
- Follow X/Twitter Developer Policy and your institution's research ethics guidelines
- Store data securely; do not share raw user data publicly

---

## Suggested Research Findings Template

```
Total tweets analysed:     ___
Positive perception:        ___%
Negative perception:        ___%
Neutral:                    ___%

Best performing ML model:  ___________
Best model accuracy:        ___%
Best model F1-score:        ___

Peak positive period:      ___________
Peak negative period:      ___________

Top positive keywords:     invest, growth, opportunity, trade, ...
Top negative keywords:     corruption, unemployment, loadshedding, ...
```

---

*Script covers all 6 CRISP-DM phases: Business Understanding →
Data Understanding → Data Preparation → Modeling → Evaluation → Deployment*
