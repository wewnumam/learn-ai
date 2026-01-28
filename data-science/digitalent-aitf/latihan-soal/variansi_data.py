def solve():
    n = int(input())
    data = []
    for _ in range(n):
        data.append(int(input()))
    #=============================== BATAS ATAS - AREA KERJA =======================
    mean = sum(data) / n
    var = 0
    for x in data:
        var += (x - mean) ** 2
    print(var / n)
    #=============================== BATAS BAWAH - AREA KERJA ======================

if __name__ == "__main__":
    solve()