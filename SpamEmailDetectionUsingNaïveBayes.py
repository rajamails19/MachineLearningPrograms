from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example dataset: Emails and their spam labels (0 = Not Spam, 1 = Spam)
emails = ["Win a free iPhone now!", "Meeting scheduled at 2 PM", "Get rich fast!", "Important update for your account"]
labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X, labels)

# Predict whether a new email is spam or not
new_email = ["You have won a lottery! Claim your prize now."]
X_new = vectorizer.transform(new_email)
prediction = model.predict(X_new)

print("Email Classification:", "Spam" if prediction[0] == 1 else "Not Spam")
