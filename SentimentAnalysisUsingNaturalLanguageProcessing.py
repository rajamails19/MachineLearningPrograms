import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = pd.DataFrame({
    "Text": ["I love this product", "This is the worst experience", "It's okay, nothing special", 
             "Absolutely amazing!", "Terrible service, not recommended"],
    "Sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative"]
})

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["Text"])
y = data["Sentiment"]

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X, y)

# Predict sentiment for a new review
new_review = ["This movie is fantastic!"]
X_new = vectorizer.transform(new_review)
prediction = model.predict(X_new)

print("Sentiment Prediction:", prediction[0])
