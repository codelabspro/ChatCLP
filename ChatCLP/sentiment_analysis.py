import torch
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
res = classifier("There is a lot of issues humankind needs to solve.")
print(res)
