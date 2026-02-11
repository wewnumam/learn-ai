import re
import string
import unicodedata
from typing import List


# ---------------------------------------------------------
# 1. Case Normalization
# ---------------------------------------------------------

def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def casefold(text: str) -> str:
    """
    Aggressive case normalization (better for multilingual text).
    Example: German ß -> ss
    """
    return text.casefold()


# ---------------------------------------------------------
# 2. Unicode Normalization
# ---------------------------------------------------------

def unicode_normalize(text: str, form: str = "NFKC") -> str:
    """
    Normalize unicode characters.

    Forms:
    - NFC  : Canonical composition
    - NFD  : Canonical decomposition
    - NFKC : Compatibility composition (most common)
    - NFKD : Compatibility decomposition
    """
    return unicodedata.normalize(form, text)


def remove_diacritics(text: str) -> str:
    """Remove accents/diacritics (é → e)."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


# ---------------------------------------------------------
# 3. Whitespace Normalization
# ---------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and trim."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------
# 4. Punctuation Normalization
# ---------------------------------------------------------

def remove_punctuation(text: str) -> str:
    """Remove all punctuation."""
    return text.translate(str.maketrans("", "", string.punctuation))


def replace_punctuation_with_space(text: str) -> str:
    """Replace punctuation with spaces."""
    return re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)


# ---------------------------------------------------------
# 5. Number Normalization
# ---------------------------------------------------------

def remove_numbers(text: str) -> str:
    """Remove all digits."""
    return re.sub(r"\d+", "", text)


def normalize_numbers(text: str, token: str = "<NUM>") -> str:
    """Replace numbers with a placeholder."""
    return re.sub(r"\d+", token, text)


# ---------------------------------------------------------
# 6. Contraction Expansion
# ---------------------------------------------------------

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(contraction, expansion, text)
    return text


# ---------------------------------------------------------
# 7. Stopword Removal
# ---------------------------------------------------------

STOPWORDS = {
    "the", "is", "in", "and", "to", "of", "a", "an", "on", "for"
}


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords from token list."""
    return [t for t in tokens if t not in STOPWORDS]


# ---------------------------------------------------------
# 8. Token Normalization
# ---------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return text.split()


def detokenize(tokens: List[str]) -> str:
    """Join tokens back into text."""
    return " ".join(tokens)


# ---------------------------------------------------------
# 9. Repeated Character Normalization
# ---------------------------------------------------------

def normalize_repeated_characters(text: str, max_repeats: int = 2) -> str:
    """
    Normalize repeated characters.
    Example: 'soooo' -> 'soo'
    """
    pattern = re.compile(r"(.)\1{" + str(max_repeats) + ",}")
    return pattern.sub(r"\1" * max_repeats, text)


# ---------------------------------------------------------
# 10. URL, Email, Mention Normalization
# ---------------------------------------------------------

def normalize_urls(text: str, token: str = "<URL>") -> str:
    return re.sub(r"https?://\S+|www\.\S+", token, text)


def normalize_emails(text: str, token: str = "<EMAIL>") -> str:
    return re.sub(r"\S+@\S+\.\S+", token, text)


def normalize_mentions(text: str, token: str = "<USER>") -> str:
    return re.sub(r"@\w+", token, text)


# ---------------------------------------------------------
# 11. HTML Normalization
# ---------------------------------------------------------

def remove_html_tags(text: str) -> str:
    """Remove HTML tags."""
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------
# 12. Full Normalization Pipeline
# ---------------------------------------------------------

def full_normalization_pipeline(text: str) -> str:
    """
    Example end-to-end normalization pipeline.
    """
    text = unicode_normalize(text)
    text = casefold(text)
    text = expand_contractions(text)
    text = normalize_urls(text)
    text = normalize_emails(text)
    text = normalize_mentions(text)
    text = remove_html_tags(text)
    text = remove_diacritics(text)
    text = normalize_repeated_characters(text)
    text = replace_punctuation_with_space(text)
    text = normalize_numbers(text)
    text = normalize_whitespace(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)

    return detokenize(tokens)


# ---------------------------------------------------------
# 13. Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":
    raw_text = """
    Héyyy!!! I can't believe THIS costs $123.45 😄😄😄
    Visit https://example.com or email me at test@example.com.
    <p>This is soooo cooool!!!</p>
    """

    print("RAW TEXT:")
    print(raw_text)

    print("\nNORMALIZED TEXT:")
    print(full_normalization_pipeline(raw_text))