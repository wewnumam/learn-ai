def solve():
    n = int(input())
    data = []
    for _ in range(n):
        data.append(int(input()))
    #=============================== BATAS ATAS - AREA KERJA =======================
    mn = min(data)
    mx = max(data)
    for x in data:
        print((x - mn) / (mx - mn))
    #=============================== BATAS BAWAH - AREA KERJA ======================

if __name__ == "__main__":
    solve()