import pandas as pd

def solve():
    n = int(input())
    rows = []
    for _ in range(n):
        name, score = input().split()
        rows.append((name, int(score)))
    #=============================== BATAS ATAS - AREA KERJA =======================
    df = pd.DataFrame(rows, columns=["name", "score"])
    df["lulus"] = df["score"] >= 75
    print(df[df["lulus"]].shape[0])
    #=============================== BATAS BAWAH - AREA KERJA ======================

if __name__ == "__main__":
    solve()