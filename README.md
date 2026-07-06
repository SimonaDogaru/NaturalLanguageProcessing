# Fact-Checking ChatGPT Outputs

This project predicts if a fact from a ChatGPT-generated biography is **supported** or **not supported** by Wikipedia passages.

## Part 1 – Word Overlap

- Tokenize fact and passage text.
- Optionally remove stopwords / lowercase.
- Compute a lexical overlap score (e.g., Jaccard or cosine over bag-of-words).
- If the best overlap score ≥ threshold → **supported**, else → **not supported**.

## Part 2 – Textual Entailment

- Split passages into sentences.
- For each (sentence, fact) pair, run a DeBERTa-v3 entailment model (premise = sentence, hypothesis = fact).
- Use the entailment / contradiction / neutral probabilities to get a single score per pair.
- Take the maximum score over all sentences; if it exceeds a threshold → **supported**, else → **not supported**.
