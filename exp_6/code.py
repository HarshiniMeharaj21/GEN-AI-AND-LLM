from transformers import pipeline

# ---------- Sentiment Analysis ----------
sentiment_analyzer = pipeline("sentiment-analysis")

reviews = [
    "The new smartphone has an amazing camera and battery life!",
    "The delivery was late and the packaging was damaged."
]

for review in reviews:
    result = sentiment_analyzer(review)[0]
    print(f"Review: {review}\n -> {result['label']} ({round(result['score'], 3)})\n")
