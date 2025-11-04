
# formation_console.py
import json, sys, re, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

JSON_PATH = Path(__file__).with_name("digit_formations_100.json")

# ---- Settings ----
Z_M = 120.0              # default altitude for reference (not used in 2D plot)
SAFETY_RADIUS_M = 2.0    # for info/debug
DWELL_SEC = 2.0          # hold each digit this many seconds
MORPH_SEC = 1.5          # morph duration between digits
FPS = 25                 # frames per second during morph
DOT_SIZE = 16            # matplotlib marker size

def load_formations(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)
    agents = data["agents"]
    digits = data["digits"]
    # build arrays per digit ordered by agent_id to ensure consistent mapping across digits
    # We'll sort by agent_id so agent i always maps to index i-1
    def points_for_digit(d):
        pts = sorted(d, key=lambda e: e["agent_id"])
        X = np.array([[e["x_norm"], e["y_norm"]] for e in pts], dtype=float)
        return X
    digit_points = {int(k): points_for_digit(v) for k,v in digits.items()}
    return agents, digit_points

def parse_digits(inp: str):
    # extract all single digits 0-9 in order of appearance
    tokens = re.findall(r"[0-9]", inp)
    if not tokens:
        raise ValueError("No digits found. Example inputs: 0  |  8,2,1  |  1 5 9")
    seq = [int(t) for t in tokens]
    return seq

def morph_sequence(ax, seq, points, dwell=DWELL_SEC, morph=MORPH_SEC, fps=FPS):
    # Initialize scatter
    first = seq[0]
    P = points[first]     # (100,2)
    scat = ax.scatter(P[:,0], P[:,1], s=DOT_SIZE)
    ax.set_title(f"Digit {first}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.5)
    plt.pause(dwell)

    for nxt in seq[1:]:
        P_next = points[nxt]
        steps = max(1, int(morph*fps))
        for k in range(1, steps+1):
            a = k/steps
            Pk = (1-a)*P + a*P_next
            scat.set_offsets(Pk)
            ax.set_title(f"Morphing {seq[seq.index(nxt)-1]} → {nxt}")
            plt.pause(1.0/fps)
        P = P_next
        scat.set_offsets(P)
        ax.set_title(f"Digit {nxt}")
        plt.pause(dwell)

def main(argv):
    # Load formations
    try:
        agents, digit_points = load_formations(JSON_PATH)
    except FileNotFoundError:
        print(f"[ERROR] JSON not found at: {JSON_PATH}")
        print("Copy 'digit_formations_100.json' next to this script and try again.")
        sys.exit(1)

    # Ask user
    if len(argv) > 1:
        raw = " ".join(argv[1:])
    else:
        raw = input("Display digit(s) (e.g., '0' or '8, 2, 1' or '1 5 9'): ").strip()
    try:
        seq = parse_digits(raw)
    except ValueError as e:
        print("[ERROR]", e)
        sys.exit(2)

    print(f"[INFO] Sequence: {seq}")
    print("[INFO] Close the window to exit.")
    # Plot
    plt.figure(figsize=(5,5))
    morph_sequence(plt.gca(), seq, digit_points, dwell=DWELL_SEC, morph=MORPH_SEC, fps=FPS)
    # keep window open until closed by user
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
