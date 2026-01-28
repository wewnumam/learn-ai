def solve():
    a, b, c, d = map(float, input().split())
    #=============================== BATAS ATAS - AREA KERJA =======================
    det = a * d - b * c
    print(f"DETERMINANT: {det}")
    print("INVERS:")
    print(d / det, -b / det)
    print(-c / det, a / det)
    #=============================== BATAS BAWAH - AREA KERJA ======================

if __name__ == "__main__":
    solve()
