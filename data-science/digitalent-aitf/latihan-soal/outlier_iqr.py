def solve():
    n = int(input())
    data = []
    for _ in range(n):
        data.append(int(input()))
    #=============================== BATAS ATAS - AREA KERJA =======================
    data.sort()
    mid = n // 2
    q1 = data[mid // 2]
    q3 = data[mid + mid // 2]
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = 0
    for x in data:
        if x < lower or x > upper:
            count += 1
    print(count)
    #=============================== BATAS BAWAH - AREA KERJA ======================

if __name__ == "__main__":
    solve()