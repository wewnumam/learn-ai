import pandas as pd

def solve():
    n = int(input())
    rows = []
    for _ in range(n):
        cat, val = input().split()
        rows.append((cat, int(val)))
    #=============================== BATAS ATAS - AREA KERJA =======================
    df = pd.DataFrame(rows, columns=["category", "value"])
    result = df.groupby("category")["value"].sum()
    for cat, total in result.items():
        print(cat, total)
    #=============================== BATAS BAWAH - AREA KERJA ======================

if __name__ == "__main__":
    solve()
