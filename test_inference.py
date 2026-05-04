#!/usr/bin/env python3
"""Test inference to verify vectorizer loads properly."""

from app import predict_sentiment

test_cases = [
    "I love this product",
    "This is terrible",
    "It's okay, nothing special",
]

print("Testing sentiment prediction...")
for text in test_cases:
    result = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Result: {result}\n")

print("✓ All tests passed!")
