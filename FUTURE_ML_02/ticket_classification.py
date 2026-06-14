import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("tickets.csv")

print(df.columns)

X = df["text"]
y = df["category"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X, y)

new_ticket = ["internet connection problem"]
new_ticket_vec = vectorizer.transform(new_ticket)

prediction = model.predict(new_ticket_vec)
print("Predicted Category:", prediction[0])