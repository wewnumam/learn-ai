import re
from collections import defaultdict

def solve():
    n = int(input())
    freq = defaultdict(int)
    #=============================== BATAS ATAS - AREA KERJA =======================
    for _ in range(n):
        text = input().lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = text.split()
        for i in range(len(tokens) - 1):
            freq[(tokens[i], tokens[i + 1])] += 1

    for k, v in freq.items():
        print(f"{k}: {v}")
    #=============================== BATAS BAWAH - AREA KERJA ======================

if __name__ == "__main__":
    solve()