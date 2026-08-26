from transformers import pipeline

# ---------- Document Classification (Zero-Shot) ----------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

document = "The central bank raised interest rates to control rising inflation."
candidate_labels = ["Politics", "Economy", "Sports", "Technology"]

classification = classifier(document, candidate_labels)
print("Document:", document)
for label, score in zip(classification["labels"], classification["scores"]):
    print(f"{label}: {round(score, 3)}")
