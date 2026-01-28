def solve():
    n = int(input())
    max_val = None
    for _ in range(n):
        x = int(input())
        #=============================== BATAS ATAS - AREA KERJA =======================
        if max_val is None or x > max_val:
            max_val = x
        #=============================== BATAS BAWAH - AREA KERJA ======================
    print(max_val)

if __name__ == "__main__":
    solve()
