def solve():
    n = int(input())
    data = []
    for _ in range(n):
        data.append(int(input()))
    #=============================== BATAS ATAS - AREA KERJA =======================
    avg = sum(data) / n
    count = 0
    for x in data:
        if x > avg:
            count += 1
    print(count)
    #=============================== BATAS BAWAH - AREA KERJA ======================

if __name__ == "__main__":
    solve()