import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -----------------------------
# Load Trained Multilingual Model
# -----------------------------

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Multilingual Sentiment Prediction
# -----------------------------

def predict_sentiment(text):
    text_lower = str(text).lower()

    negative_words = [
        "worst", "bad", "damaged", "delay",
        "rude", "chetta", "bekaar", "bekar",
        "kharab", "kharap", "baje"
    ]

    positive_words = [
        "excellent", "super", "bagundi",
        "achha", "mast", "bhalo",
        "awesome", "darun"
    ]

    neg_hits = sum(word in text_lower for word in negative_words)
    pos_hits = sum(word in text_lower for word in positive_words)

    vec = vectorizer.transform([text_lower])
    model_prediction = model.predict(vec)[0]

    # Lexicon override logic
    if neg_hits > pos_hits:
        return "negative"
    elif pos_hits > neg_hits:
        return "positive"
    else:
        return model_prediction


# -----------------------------
# Multilingual Industry Dictionaries
# -----------------------------

industry_dictionaries = {

    "Restaurant": {
        "Service Issues": [
            "slow", "late", "rude",
            "service slow", "late ayindi",
            "late tha", "late chilo"
        ],
        "Food Quality": [
            "cold", "bad taste", "chetta",
            "bekaar", "kharap"
        ],
        "Pricing": [
            "expensive", "ekkuva", "zyada", "beshi"
        ]
    },

    "E-commerce": {
        "Delivery Problems": [
            "late delivery", "delay", "late",
            "late ayindi", "late tha", "late chilo"
        ],
        "Product Quality": [
            "damaged", "broken", "chetta",
            "bekaar", "kharap"
        ],
        "Customer Support": [
            "no response", "reply nahi",
            "reply koreni", "rude support"
        ]
    },

    "Hospital": {
        "Staff Behavior": [
            "rude", "careless",
            "rude tha", "rude chilo"
        ],
        "Waiting Time": [
            "long wait", "delay", "ekkuva",
            "zyada", "beshi"
        ],
        "Billing Issues": [
            "overcharged", "hidden charges",
            "expensive", "zyada", "beshi"
        ]
    },

    "Movies": {
        "Story Issues": [
            "boring", "weak story",
            "weak thi", "weak chilo"
        ],
        "Length Issues": [
            "too long", "length ekkuva",
            "lambi thi", "lamba chilo"
        ],
        "Acting Issues": [
            "bad acting", "acting worst",
            "bekaar acting", "kharap acting"
        ]
    }
}

def detect_business_issues(texts, industry):
    selected = industry_dictionaries[industry]
    issue_count = {k: 0 for k in selected}

    for text in texts:
        text_lower = text.lower()
        for category, keywords in selected.items():
            for word in keywords:
                if word in text_lower:
                    issue_count[category] += 1

    return issue_count

# -----------------------------
# Intelligent Recommendation Engine
# -----------------------------

def generate_recommendations(issue_count, sentiment_counts, total_reviews):

    recommendations = []
    negative_count = sentiment_counts.get("negative", 0)
    negative_ratio = (negative_count / total_reviews) * 100 if total_reviews > 0 else 0

    sorted_issues = sorted(issue_count.items(), key=lambda x: x[1], reverse=True)

    for issue, count in sorted_issues:
        if count == 0:
            continue

        issue_ratio = (count / total_reviews) * 100

        if issue_ratio > 25:
            severity = "Critical"
        elif issue_ratio > 15:
            severity = "High"
        elif issue_ratio > 8:
            severity = "Moderate"
        else:
            severity = "Low"

        insight = (
            f"{severity} priority: '{issue}' appears in {round(issue_ratio,2)}% "
            f"of reviews. Strategic action recommended."
        )

        recommendations.append(insight)

    if negative_ratio > 40:
        recommendations.append(
            f"Negative sentiment at {round(negative_ratio,2)}%. "
            f"Brand perception risk is high."
        )
    elif negative_ratio > 25:
        recommendations.append(
            f"Negative sentiment at {round(negative_ratio,2)}%. "
            f"Targeted operational improvements suggested."
        )
    else:
        recommendations.append(
            f"Overall sentiment stable with {round(negative_ratio,2)}% negative feedback."
        )

    return recommendations

# -----------------------------
# Executive Summary
# -----------------------------

def generate_summary(sentiment_counts, issue_count, total_reviews):

    pos = sentiment_counts.get("positive", 0)
    neg = sentiment_counts.get("negative", 0)
    neu = sentiment_counts.get("neutral", 0)

    pos_ratio = round((pos / total_reviews) * 100, 2) if total_reviews > 0 else 0
    neg_ratio = round((neg / total_reviews) * 100, 2) if total_reviews > 0 else 0
    neu_ratio = round((neu / total_reviews) * 100, 2) if total_reviews > 0 else 0

    dominant_issue = max(issue_count, key=issue_count.get) if issue_count else None

    summary = (
        f"Out of {total_reviews} reviews analyzed, "
        f"{pos_ratio}% positive, {neg_ratio}% negative, "
        f"{neu_ratio}% neutral sentiment detected. "
    )

    if dominant_issue and issue_count[dominant_issue] > 0:
        summary += f"Primary operational concern identified in '{dominant_issue}'."
    else:
        summary += "No dominant operational concern detected."

    return summary

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌍 Pan-India Code-Mixed Review Intelligence Dashboard")
st.write("Telugu-English | Hindi-English | Bengali-English")

industry = st.selectbox(
    "Select Industry",
    ["Restaurant", "E-commerce", "Hospital", "Movies"]
)

uploaded_file = st.file_uploader("Upload CSV with column name 'review'", type=["csv"])
single_review = st.text_area("Or type a single review")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("CSV must contain column named 'review'")
    else:
        df["Sentiment"] = df["review"].apply(predict_sentiment)

        st.subheader("📊 Sentiment Breakdown")
        sentiment_counts = df["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

        st.subheader("🔎 Detailed Results")
        st.dataframe(df)

        issues = detect_business_issues(df["review"], industry)

        st.subheader("📌 Business Issues")
        issue_df = pd.DataFrame(list(issues.items()), columns=["Category", "Count"])
        st.bar_chart(issue_df.set_index("Category"))

        total_reviews = len(df)

        recommendations = generate_recommendations(
            issues, sentiment_counts, total_reviews
        )

        st.subheader("💡 Strategic Insights")
        for r in recommendations:
            st.write("•", r)

        summary_text = generate_summary(
            sentiment_counts, issues, total_reviews
        )

        st.subheader("🧠 Executive Summary")
        st.write(summary_text)

if single_review:
    sentiment = predict_sentiment(single_review)
    st.subheader("Prediction Result")
    st.write(f"Sentiment: {sentiment}")

