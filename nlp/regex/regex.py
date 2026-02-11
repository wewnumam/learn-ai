"""
regex_examples.py

A comprehensive, structured collection of regular expression examples in Python.
Each section builds on core regex principles and includes executable examples.
"""

import re

# --------------------------------------------------
# 1. BASIC LITERALS
# --------------------------------------------------
print("\n--- BASIC LITERALS ---")

text = "hello world"
print(re.search("hello", text))        # Match literal
print(re.search("world", text))
print(re.search("Hello", text))        # Case-sensitive (None)

# --------------------------------------------------
# 2. CHARACTER CLASSES
# --------------------------------------------------
print("\n--- CHARACTER CLASSES ---")

text = "cat bat rat mat"
print(re.findall("[cb]at", text))      # c or b
print(re.findall("[a-z]at", text))     # any lowercase letter
print(re.findall("[^b]at", text))      # NOT b

# --------------------------------------------------
# 3. PREDEFINED CHARACTER SETS
# --------------------------------------------------
print("\n--- PREDEFINED CHARACTER SETS ---")

text = "User123 email@test.com"
print(re.findall(r"\d", text))         # Digits
print(re.findall(r"\D", text))         # Non-digits
print(re.findall(r"\w", text))         # Word characters
print(re.findall(r"\W", text))         # Non-word
print(re.findall(r"\s", text))         # Whitespace
print(re.findall(r"\S", text))         # Non-whitespace

# --------------------------------------------------
# 4. QUANTIFIERS
# --------------------------------------------------
print("\n--- QUANTIFIERS ---")

text = "aaa aa a"
print(re.findall("a*", text))          # 0 or more
print(re.findall("a+", text))          # 1 or more
print(re.findall("a?", text))          # 0 or 1
print(re.findall("a{2}", text))        # Exactly 2
print(re.findall("a{1,3}", text))      # Between 1 and 3

# --------------------------------------------------
# 5. GREEDY vs LAZY
# --------------------------------------------------
print("\n--- GREEDY vs LAZY ---")

text = "<tag>content</tag>"
print(re.findall("<.*>", text))        # Greedy
print(re.findall("<.*?>", text))       # Lazy

# --------------------------------------------------
# 6. ANCHORS
# --------------------------------------------------
print("\n--- ANCHORS ---")

text = "start middle end"
print(re.search("^start", text))       # Beginning
print(re.search("end$", text))          # End

# --------------------------------------------------
# 7. GROUPS
# --------------------------------------------------
print("\n--- GROUPS ---")

text = "2025-02-10"
match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
print(match.groups())
print(match.group(1))                  # Year

# --------------------------------------------------
# 8. NON-CAPTURING GROUPS
# --------------------------------------------------
print("\n--- NON-CAPTURING GROUPS ---")

text = "cat bat rat"
print(re.findall(r"(?:c|b)at", text))

# --------------------------------------------------
# 9. ALTERNATION (OR)
# --------------------------------------------------
print("\n--- ALTERNATION ---")

text = "apple banana orange"
print(re.findall("apple|orange", text))

# --------------------------------------------------
# 10. LOOKAHEADS
# --------------------------------------------------
print("\n--- LOOKAHEADS ---")

text = "password123"
print(re.findall(r"\w+(?=\d)", text))  # Followed by digit
print(re.findall(r"\w+(?!\d)", text))  # Not followed by digit

# --------------------------------------------------
# 11. LOOKBEHINDS
# --------------------------------------------------
print("\n--- LOOKBEHINDS ---")

text = "$100 €200"
print(re.findall(r"(?<=\$)\d+", text)) # After $
print(re.findall(r"(?<!\$)\d+", text)) # Not after $

# --------------------------------------------------
# 12. ESCAPING SPECIAL CHARACTERS
# --------------------------------------------------
print("\n--- ESCAPING ---")

text = "3.14 + 2.71"
print(re.findall(r"\d+\.\d+", text))   # Escape dot

# --------------------------------------------------
# 13. FLAGS
# --------------------------------------------------
print("\n--- FLAGS ---")

text = "Hello\nWorld"
print(re.findall("^world", text, re.IGNORECASE | re.MULTILINE))

# --------------------------------------------------
# 14. SPLIT
# --------------------------------------------------
print("\n--- SPLIT ---")

text = "one,two;three four"
print(re.split("[,; ]", text))

# --------------------------------------------------
# 15. SUBSTITUTE
# --------------------------------------------------
print("\n--- SUBSTITUTE ---")

text = "My phone is 123-456-7890"
print(re.sub(r"\d", "X", text))

# --------------------------------------------------
# 16. VALIDATION PATTERNS
# --------------------------------------------------
print("\n--- VALIDATION ---")

# Email
email = "test.user@example.com"
print(bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email)))

# Phone number
phone = "123-456-7890"
print(bool(re.fullmatch(r"\d{3}-\d{3}-\d{4}", phone)))

# --------------------------------------------------
# 17. REAL-WORLD EXTRACTION
# --------------------------------------------------
print("\n--- REAL-WORLD ---")

text = "Visit https://example.com or http://test.org"
print(re.findall(r"https?://\S+", text))

text = "Prices: $10, $20, $30"
print(re.findall(r"\$\d+", text))

# --------------------------------------------------
# 18. COMPILED REGEX
# --------------------------------------------------
print("\n--- COMPILED REGEX ---")

pattern = re.compile(r"\b\w{4}\b")
text = "This test finds four letter words"
print(pattern.findall(text))

# --------------------------------------------------
# END
# --------------------------------------------------