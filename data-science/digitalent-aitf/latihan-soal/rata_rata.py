def solve():
    n = int(input())
    total = 0
    for _ in range(n):
        total += int(input())
    #=============================== BATAS ATAS - AREA KERJA =======================
    print(total / n)
    #=============================== BATAS BAWAH - AREA KERJA ======================

if __name__ == "__main__":
    solve()