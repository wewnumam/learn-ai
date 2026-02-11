"""
Text Normalization From Scratch (No NLP Libraries)

Covers:
1. Tokenization
2. Stemming (rule-based)
3. Lemmatization (rule-based)
4. Byte Pair Encoding (BPE)

Dependencies:
- Python 3.x
- re (standard library only)
"""

import re
from collections import Counter, defaultdict


# =========================================================
# 1. TOKENIZATION (Rule-based)
# =========================================================

def tokenize(text: str):
    """
    Basic word tokenization:
    - lowercase
    - split on non-alphanumeric characters
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


# =========================================================
# 2. STEMMING (Simplified Porter-style)
# =========================================================

def stem(word: str):
    """
    Very simplified English stemmer.
    Demonstrates suffix stripping logic.
    """

    rules = [
        ("sses", "ss"),
        ("ies", "i"),
        ("ss", "ss"),
        ("s", ""),
        ("ing", ""),
        ("ed", ""),
        ("ly", ""),
    ]

    for suffix, replacement in rules:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)] + replacement

    return word


def stem_tokens(tokens):
    return [stem(t) for t in tokens]


# =========================================================
# 3. LEMMATIZATION (Rule-based + Dictionary)
# =========================================================

LEMMA_DICT = {
    "am": "be",
    "is": "be",
    "are": "be",
    "was": "be",
    "were": "be",
    "running": "run",
    "ran": "run",
    "cars": "car",
    "children": "child",
    "better": "good",
}


def lemmatize(word: str):
    """
    Rule + dictionary-based lemmatization.
    """

    if word in LEMMA_DICT:
        return LEMMA_DICT[word]

    # plural noun heuristic
    if word.endswith("s") and len(word) > 3:
        return word[:-1]

    # verb heuristic
    if word.endswith("ing"):
        return word[:-3]

    if word.endswith("ed"):
        return word[:-2]

    return word


def lemmatize_tokens(tokens):
    return [lemmatize(t) for t in tokens]


# =========================================================
# 4. BYTE PAIR ENCODING (BPE)
# =========================================================

def get_stats(vocab):
    """
    Count frequency of symbol pairs.
    """
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    """
    Merge most frequent pair in vocabulary.
    """
    merged_vocab = {}
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(rf"(?<!\S){bigram}(?!\S)")

    for word in vocab:
        new_word = pattern.sub("".join(pair), word)
        merged_vocab[new_word] = vocab[word]

    return merged_vocab


def train_bpe(corpus, num_merges=10):
    """
    Minimal BPE training loop.
    """
    vocab = Counter()

    for word in corpus:
        vocab[" ".join(list(word)) + " </w>"] += 1

    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)

    return vocab


def bpe_encode(word, bpe_vocab):
    """
    Encode word using learned BPE merges.
    """
    symbols = list(word) + ["</w>"]
    while True:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        mergeable = None

        for pair in pairs:
            candidate = " ".join(pair)
            for v in bpe_vocab:
                if candidate.replace(" ", "") in v.replace(" ", ""):
                    mergeable = pair
                    break
            if mergeable:
                break

        if not mergeable:
            break

        i = symbols.index(mergeable[0])
        symbols[i:i + 2] = ["".join(mergeable)]

    return symbols


# =========================================================
# 5. DEMO
# =========================================================

if __name__ == "__main__":

    text = "The children were running quickly with cars"

    print("RAW TEXT:")
    print(text)
    print()

    # Tokenization
    tokens = tokenize(text)
    print("TOKENS:")
    print(tokens)
    print()

    # Stemming
    stemmed = stem_tokens(tokens)
    print("STEMMED:")
    print(stemmed)
    print()

    # Lemmatization
    lemmatized = lemmatize_tokens(tokens)
    print("LEMMATIZED:")
    print(lemmatized)
    print()

    # BPE
    print("BPE TRAINING:")
    corpus = tokens
    bpe_vocab = train_bpe(corpus, num_merges=10)
    print("Learned BPE Vocabulary:")
    for k in list(bpe_vocab.keys())[:5]:
        print(k)
    print()

    print("BPE ENCODING (example: 'running'):")
    print(bpe_encode("running", bpe_vocab))
