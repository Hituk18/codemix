import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load all datasets
df_telugu = pd.read_csv("telugu_english_multidomain_emoji_3000_1.csv")
df_hindi = pd.read_csv("hindi_english_multidomain_emoji_3000.csv")
df_bengali = pd.read_csv("bengali_english_multidomain_emoji_3000.csv")

# Combine
df = pd.concat([df_telugu, df_hindi, df_bengali], ignore_index=True)

df["text"] = df["text"].str.lower()

# Character TF-IDF
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    min_df=3
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=4000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Unified multilingual model trained successfully.")
