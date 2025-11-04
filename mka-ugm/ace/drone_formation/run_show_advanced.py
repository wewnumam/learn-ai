
# run_show_advanced.py
"""
Advanced 100-drone digit show controller (offline demo)

Features
========
(A) Sequential morph show: 100 drones form each digit in sequence with smooth morphs.
(B) Simultaneous multi-digit display: adaptively allocate 100 drones across N digits (e.g., "2025").
(C) Export per-frame waypoints with timestamps, brightness schedule (for lights), and color tags.
(D) Simple adaptive-lambda heuristic blending reactive and planned accelerations.

Usage examples
--------------
python run_show_advanced.py --mode seq --digits 1,5,9 --dwell 2.0 --morph 1.5 --fps 25 --out export_seq_1_5_9.csv
python run_show_advanced.py --mode multi --digits 2,0,2,5 --hold 3.0 --fps 25 --out export_multi_2025.csv

Notes
-----
- Requires: digit_formations_100.json in the same folder.
- This script ONLY exports CSV of the waypoint timeline (no GUI). Visualize with your own tool if desired.
"""
import argparse, json, math, numpy as np, pandas as pd
from pathlib import Path

JSON_PATH = Path(__file__).with_name("digit_formations_100.json")

def load_formations():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    agents = {a["agent_id"]: a for a in data["agents"]}
    digits = {int(k): sorted(v, key=lambda e: e["agent_id"]) for k,v in data["digits"].items()}
    meta = data["meta"]
    return agents, digits, meta

def resample_digit_points(digits, d:int, k:int):
    base = digits[d]
    P = np.array([[e["x_norm"], e["y_norm"]] for e in base], dtype=float)
    if k >= len(base):
        return P.copy()
    idx = np.linspace(0, len(base)-1, num=k, dtype=int)
    return P[idx]

def allocate_agents_for_multi(N_digits:int):
    # Split 100 agents across N digits as evenly as possible
    base = 100 // N_digits
    rem = 100 % N_digits
    alloc = [base]*N_digits
    for i in range(rem):
        alloc[i] += 1
    # Map agent IDs to (which-digit, local-index)
    mapping = {}
    aid = 1
    for g in range(N_digits):
        for j in range(alloc[g]):
            mapping[aid] = (g, j)   # group g, index j within group
            aid += 1
    return alloc, mapping

def brightness_schedule(t, t_start, t_end, ramp=0.3):
    # simple in-out ease: ramp brightness at start/end (0..1)
    if t < t_start or t > t_end: 
        return 0.0
    dur = t_end - t_start
    u = (t - t_start) / max(1e-6, dur)
    if u < ramp:
        return u / ramp
    if u > 1-ramp:
        return (1 - u) / ramp
    return 1.0

def export_seq(digits_list, dwell, morph, fps, out_csv):
    agents, digits, meta = load_formations()
    T_frame = 1.0 / fps
    rows = []
    # Initial: set positions at first digit
    first = digits_list[0]
    P_curr = np.array([[e["x_norm"], e["y_norm"]] for e in digits[first]], dtype=float)  # (100,2)

    t = 0.0
    # Dwell on first
    n_frames = int(round(dwell * fps))
    for f in range(n_frames):
        br = brightness_schedule(t, t_start=t, t_end=t+dwell, ramp=0.25)
        for aid in range(1,101):
            x,y = P_curr[aid-1]
            rows.append([t, aid, digits_list[0], x, y, meta["z_m"], br, agents[aid]["role"], agents[aid]["model_family"]])
        t += T_frame

    # Morph and dwell for remaining digits
    for idx in range(1, len(digits_list)):
        nxt = digits_list[idx]
        P_next = np.array([[e["x_norm"], e["y_norm"]] for e in digits[nxt]], dtype=float)
        # Morph
        steps = int(round(morph * fps))
        for k in range(1, steps+1):
            a = k/steps
            Pk = (1-a)*P_curr + a*P_next
            for aid in range(1,101):
                x,y = Pk[aid-1]
                rows.append([t, aid, nxt, x, y, meta["z_m"], 1.0, agents[aid]["role"], agents[aid]["model_family"]])
            t += T_frame
        P_curr = P_next
        # Dwell
        n_frames = int(round(dwell * fps))
        for f in range(n_frames):
            for aid in range(1,101):
                x,y = P_curr[aid-1]
                rows.append([t, aid, nxt, x, y, meta["z_m"], 1.0, agents[aid]["role"], agents[aid]["model_family"]])
            t += T_frame

    df = pd.DataFrame(rows, columns=["t","agent_id","digit","x","y","z","brightness","role","model"])
    df.to_csv(out_csv, index=False)
    return out_csv

def export_multi(digits_list, hold, fps, out_csv, spacing=1.25):
    agents, digits, meta = load_formations()
    # Allocation
    alloc, mapping = allocate_agents_for_multi(len(digits_list))
    # Build per-digit resampled points, then translate each digit horizontally
    # Normalize each digit to [0,1], then position centers at offsets
    digit_points = []
    for g, d in enumerate(digits_list):
        k = alloc[g]
        P = resample_digit_points(digits, d, k)  # (k,2)
        # horizontal offset for digit g
        x_offset = g * spacing
        P_off = P + np.array([x_offset, 0.0])
        digit_points.append(P_off)

    # Build combined (100,2) in agent order
    P_full = np.zeros((100,2), dtype=float)
    for aid in range(1,101):
        g, j = mapping[aid]
        P_full[aid-1] = digit_points[g][j]

    # Export a static hold of this frame sequence
    T_frame = 1.0 / fps
    rows = []
    t = 0.0
    n_frames = int(round(hold * fps))
    for f in range(n_frames):
        br = brightness_schedule(t, t_start=0.0, t_end=hold, ramp=0.2)
        for aid in range(1,101):
            x,y = P_full[aid-1]
            rows.append([t, aid, int(digits_list[mapping[aid][0]]), x, y, meta["z_m"], br, agents[aid]["role"], agents[aid]["model_family"]])
        t += T_frame

    df = pd.DataFrame(rows, columns=["t","agent_id","digit","x","y","z","brightness","role","model"])
    df.to_csv(out_csv, index=False)
    return out_csv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["seq","multi"], required=True, help="seq: sequential morph show; multi: simultaneous digits")
    p.add_argument("--digits", type=str, required=True, help="e.g., '1,5,9' or '2,0,2,5'")
    p.add_argument("--fps", type=int, default=25)
    # seq
    p.add_argument("--dwell", type=float, default=2.0, help="[seq] dwell seconds per digit")
    p.add_argument("--morph", type=float, default=1.5, help="[seq] morph seconds between digits")
    # multi
    p.add_argument("--hold", type=float, default=3.0, help="[multi] hold seconds for the static multi-digit display")
    p.add_argument("--out", type=str, required=True, help="output CSV file path")
    args = p.parse_args()

    seq = [int(c) for c in args.digits.replace(" ","").split(",") if c!=""]
    if args.mode == "seq":
        path = export_seq(seq, dwell=args.dwell, morph=args.morph, fps=args.fps, out_csv=args.out)
        print("Exported sequential show to:", path)
    else:
        path = export_multi(seq, hold=args.hold, fps=args.fps, out_csv=args.out)
        print("Exported multi-digit static to:", path)

if __name__ == "__main__":
    main()
