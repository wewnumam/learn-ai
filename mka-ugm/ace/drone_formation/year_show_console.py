
# year_show_console.py
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

JSON_PATH = Path(__file__).with_name("digit_formations_100.json")

# ---- Parameters you can tweak ----
DWELL_SEC   = 2.0   # waktu diam per digit (sequential)
MORPH_SEC   = 1.5   # durasi morph antar digit (sequential)
FPS         = 25    # frame rate animasi
DOT_SIZE    = 18    # ukuran titik drone
SPACING     = 1.25  # jarak horizontal antar digit untuk mode simultan (dalam skala normal [0..1])

def load_formations():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    agents = {a["agent_id"]: a for a in data["agents"]}
    # urutkan per digit berdasarkan agent_id supaya konsisten
    digits = {int(k): sorted(v, key=lambda e: e["agent_id"]) for k,v in data["digits"].items()}
    return agents, digits, data["meta"]

def parse_year_input(raw: str):
    # ambil semua digit 0-9 dari masukan
    s = "".join(re.findall(r"\d", raw))
    if not s:
        raise ValueError("Input tidak mengandung digit. Contoh: 2025, 3020, 4180.")
    return [int(c) for c in s]

def morph_sequence(ax, seq, points, dwell=DWELL_SEC, morph=MORPH_SEC, fps=FPS):
    # Tampilkan digit pertama
    first = seq[0]
    P = np.array([[e["x_norm"], e["y_norm"]] for e in points[first]], dtype=float)  # (100,2)
    scat = ax.scatter(P[:,0], P[:,1], s=DOT_SIZE)
    ax.set_title(f"Tahun: {''.join(map(str, seq))}  |  Digit {first}", fontsize=11)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.5)
    plt.pause(dwell)

    # Morph ke digit-digit berikutnya
    for idx in range(1, len(seq)):
        nxt = seq[idx]
        P_next = np.array([[e["x_norm"], e["y_norm"]] for e in points[nxt]], dtype=float)
        steps = max(1, int(morph*fps))
        for k in range(1, steps+1):
            a = k/steps
            Pk = (1-a)*P + a*P_next
            scat.set_offsets(Pk)
            ax.set_title(f"Tahun: {''.join(map(str, seq))}  |  Morph {seq[idx-1]} → {nxt}", fontsize=11)
            plt.pause(1.0/fps)
        P = P_next
        scat.set_offsets(P)
        ax.set_title(f"Tahun: {''.join(map(str, seq))}  |  Digit {nxt}", fontsize=11)
        plt.pause(dwell)

def show_simultaneous(ax, seq, points, spacing=SPACING, fps=FPS, hold=3.0):
    # Bagi 100 drone secara merata ke setiap digit
    N = 100
    base = N // len(seq)
    rem  = N % len(seq)
    alloc = [base]*len(seq)
    for i in range(rem):
        alloc[i] += 1
    # siapkan target untuk tiap digit
    digit_arrays = []
    for g, d in enumerate(seq):
        P = np.array([[e["x_norm"], e["y_norm"]] for e in points[d]], dtype=float)
        k = alloc[g]
        if k < len(P):
            idx = np.linspace(0, len(P)-1, num=k, dtype=int)
            P = P[idx]
        # geser secara horizontal
        P = P + np.array([g*spacing, 0.0])
        digit_arrays.append(P)
    # gabungkan sesuai urutan agent_id 1..100
    P_full = np.zeros((N,2), dtype=float)
    aid = 0
    for g, P in enumerate(digit_arrays):
        for j in range(P.shape[0]):
            P_full[aid] = P[j]; aid += 1
    # tampilkan
    scat = ax.scatter(P_full[:,0], P_full[:,1], s=DOT_SIZE)
    ax.set_title(f"Tahun: {''.join(map(str, seq))}  |  Mode simultan (hold {hold:.1f}s)", fontsize=11)
    ax.set_xlim(-0.2, (len(seq)-1)*spacing + 1.2)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.5)
    # hold beberapa detik (fps * hold frames)
    frames = int(hold*fps)
    for _ in range(max(1, frames)):
        plt.pause(1.0/fps)

def main(argv):
    try:
        agents, digits, meta = load_formations()
    except FileNotFoundError:
        print("[ERROR] File 'digit_formations_100.json' tidak ditemukan di folder yang sama.")
        sys.exit(1)

    # 1) Input tahun
    raw_year = input("Masukkan tahunnya (contoh: 2025 / 3020 / 4180): ").strip()
    try:
        seq = parse_year_input(raw_year)
    except ValueError as e:
        print("[ERROR]", e)
        sys.exit(2)

    # 2) Pilih mode
    raw_mode = input("Pilih mode (ketik 'seq' untuk berurutan/morph, atau 'multi' untuk tampil bersamaan): ").strip().lower()
    if raw_mode not in ("seq","multi"):
        print("[INFO] Mode tidak dikenali, default ke 'seq'.")
        raw_mode = "seq"

    # 3) Tampilkan ke layar
    plt.figure(figsize=(6,5))
    ax = plt.gca()
    if raw_mode == "seq":
        morph_sequence(ax, seq, digits, dwell=DWELL_SEC, morph=MORPH_SEC, fps=FPS)
    else:
        show_simultaneous(ax, seq, digits, spacing=SPACING, fps=FPS, hold=3.0)

    print("[INFO] Tutup jendela untuk mengakhiri.")
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
