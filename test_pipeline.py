from transformers import pipeline

qa = pipeline("text2text-generation", model="google/flan-t5-small")
print(qa("What is AI?", max_new_tokens=50))