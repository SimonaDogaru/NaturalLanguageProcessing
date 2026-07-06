# factcheck.py
# pyrefly: ignore [missing-import]
import torch
from typing import List
import numpy as np
import spacy
import gc
import nltk
from nltk.corpus import stopwords
import string


class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Move inputs to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Labels for MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli are:
        # 0 -> entailment, 1 -> neutral, 2 -> contradiction
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        entailment_prob = float(probs[0])

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits, probs
        gc.collect()

        return entailment_prob


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


# pyrefly: ignore [missing-import]
from nltk.stem import PorterStemmer

class WordRecallThresholdFactChecker(object):
    def __init__(self, threshold: float = 0.61):
        self.threshold = threshold
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [self.stemmer.stem(t) for t in tokens if t not in string.punctuation and t not in self.stop_words]
        return tokens

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = set(self.tokenize(fact))
        if not fact_tokens:
            return "S"

        # Combine all passages for word recall calculation or take max over passages?
        # "A supported fact might be expected to have high unigram overlap with a passage that supports it". = >
        # This implies max over passages
        max_recall = 0.0
        for passage in passages:
            passage_tokens = set(self.tokenize(passage['text']))
            if not passage_tokens:
                continue
            recall = len(fact_tokens.intersection(passage_tokens)) / len(fact_tokens)
            if recall > max_recall:
                max_recall = recall
        
        return "S" if max_recall >= self.threshold else "NS"


class EntailmentFactChecker(object):
    def __init__(self, ent_model, overlap_threshold: float =0.15, entailment_threshold: float = 0.60):
        self.ent_model = ent_model
        self.overlap_threshold = overlap_threshold
        self.entailment_threshold = entailment_threshold
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def _tokenize_for_overlap(self, text: str) -> set:
        """Tokenize text for word overlap pruning."""
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = {self.stemmer.stem(t) for t in tokens if t not in string.punctuation and t not in self.stop_words}
        return tokens

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split passage text into individual sentences.
        Passages use <s>...</s> tags around sentences from Wikipedia.
        Falls back to NLTK sentence tokenizer if no tags found.
        """
        import re
        # Extract text between <s> and </s> tags
        sentences = re.findall(r'<s>(.*?)</s>', text, re.DOTALL)
        if not sentences:
            # Fallback: use NLTK sentence tokenizer
            sentences = nltk.sent_tokenize(text)

        cleaned = []
        for sent in sentences:
            # Clean up whitespace
            sent = sent.strip()
            # Remove section headers (short text ending with a period-free pattern like "Career." or "Personal life.")
            # Keep only substantive sentences (at least 5 words)
            if len(sent.split()) < 4:
                continue
            # Skip sentences that are too long (likely noisy)
            if len(sent.split()) > 200:
                continue
            cleaned.append(sent)
        return cleaned

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = self._tokenize_for_overlap(fact)
        max_entailment_score = 0.0

        for passage in passages:
            sentences = self._split_sentences(passage['text'])
            for sentence in sentences:
                # Word overlap pruning: skip sentences with very low overlap
                sent_tokens = self._tokenize_for_overlap(sentence)
                if not fact_tokens or not sent_tokens:
                    continue
                overlap = len(fact_tokens.intersection(sent_tokens)) / len(fact_tokens)
                if overlap < self.overlap_threshold:
                    continue

                # Run entailment: sentence is premise, fact is hypothesis
                entailment_score = self.ent_model.check_entailment(sentence, fact)
                
                if entailment_score > max_entailment_score:
                    max_entailment_score = entailment_score

                # Early exit if we find strong entailment
                if max_entailment_score > 0.95:
                    break
            if max_entailment_score > 0.95:
                break

        return "S" if max_entailment_score >= self.entailment_threshold else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

