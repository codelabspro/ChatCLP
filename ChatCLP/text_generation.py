import torch

from transformers import pipeline
generator = pipeline("text-generation", model="distilgpt2")
res = generator(
    "I am the author Lewis Carrol. And the meaning of the word Jabberwocky is ",
    max_length=300,
    num_return_sequences=2,
)
print(res)
