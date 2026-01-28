import numpy as np

def solve():
    n = int(input())
    data = []
    for _ in range(n):
        data.append(int(input()))
    #=============================== BATAS ATAS - AREA KERJA =======================
    arr = np.array(data)
    print(arr.mean())
    print(np.median(arr))
    print(arr.std())
    #=============================== BATAS BAWAH - AREA KERJA ======================

if __name__ == "__main__":
    solve()
