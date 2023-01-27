import torch

from transformers import pipeline
classifier = pipeline("zero-shot-classification")
res = classifier(
    "It was the best of times. It was the worst of times",
    candidate_labels=["science fiction", "satire", "documentary", "economics", "politics", "social science"]
)
print(res)
