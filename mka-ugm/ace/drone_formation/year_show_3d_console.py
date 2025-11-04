
# year_show_3d_console.py
import json, re, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

JSON_PATH = Path(__file__).with_name("digit_formations_100.json")

# ===== User-tweakable defaults =====
DEFAULT_MODE      = "seq"   # 'seq' or 'multi'
DEFAULT_DWELL     = 2.0     # seconds per digit (seq mode)
DEFAULT_MORPH     = 1.5     # seconds between digits (seq mode)
DEFAULT_HOLD      = 3.0     # seconds hold (multi mode)
DEFAULT_FPS       = 25      # frames per second
DEFAULT_DOT_SIZE  = 18      # matplotlib marker size
DEFAULT_SPACING   = 1.25    # horizontal spacing (simultaneous mode, normalized units)

# Dynamics / safety (used in both modes)
V_MAX   = 8.0      # m/s
A_MAX   = 3.5      # m/s^2
R_SAFE  = 2.0      # m (desired min planar separation)
R_ACT   = 3.5      # m (activation distance for repulsion)
K_P     = 3.0      # PD position gain
K_V     = 2.0      # PD velocity damper
K_REP   = 12.0     # repulsion strength
LAM0    = 0.6      # initial lambda (reactive/planned blend)
LAM_UP  = 0.05     # step up when near violation
LAM_DN  = 0.01     # step down when safe
LAM_MIN = 0.3
LAM_MAX = 1.0

def load_formations():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    agents = {a["agent_id"]: a for a in data["agents"]}
    digits = {int(k): sorted(v, key=lambda e: e["agent_id"]) for k,v in data["digits"].items()}
    meta   = data["meta"]
    return agents, digits, meta

def parse_year(s: str):
    ds = re.findall(r"\d", s)
    if not ds:
        raise ValueError("Input tidak mengandung digit 0-9.")
    return [int(c) for c in ds]

def resample_digit_points(digits, d, k):
    P = np.array([[e["x_norm"], e["y_norm"]] for e in digits[d]], dtype=float)
    if k >= len(P): return P.copy()
    idx = np.linspace(0, len(P)-1, num=k, dtype=int)
    return P[idx]

def allocate_agents_for_multi(N_digits: int):
    base = 100 // N_digits
    rem  = 100 % N_digits
    alloc = [base]*N_digits
    for i in range(rem): alloc[i] += 1
    mapping = {}
    aid = 1
    for g in range(N_digits):
        for j in range(alloc[g]):
            mapping[aid] = (g, j); aid += 1
    return alloc, mapping

def build_targets_seq(digits, seq, dwell, morph):
    """Return (P_list, t_list) where P_list is list of (100,2) and t_list is list of (t0,t1)."""
    P_list, t_list = [], []
    t0 = 0.0
    # first digit dwell
    P_list.append(np.array([[e["x_norm"], e["y_norm"]] for e in digits[seq[0]]], dtype=float))
    t_list.append((t0, t0 + dwell)); t0 += dwell
    for d in seq[1:]:
        # morph segment target (linearly interpolated later)
        P_list.append(np.array([[e["x_norm"], e["y_norm"]] for e in digits[d]], dtype=float))
        t_list.append((t0, t0 + morph)); t0 += morph
        # dwell on new digit
        P_list.append(np.array([[e["x_norm"], e["y_norm"]] for e in digits[d]], dtype=float))
        t_list.append((t0, t0 + dwell)); t0 += dwell
    return P_list, t_list

def build_targets_multi(digits, seq, spacing=DEFAULT_SPACING):
    """Return single (100,2) field with all digits side-by-side (100 drones split)."""
    alloc, mapping = allocate_agents_for_multi(len(seq))
    digit_arrays = []
    for g, d in enumerate(seq):
        k = alloc[g]
        P = resample_digit_points(digits, d, k)
        P = P + np.array([g*spacing, 0.0])  # horizontal offset
        digit_arrays.append(P)
    P_full = np.zeros((100,2), dtype=float)
    for aid in range(1,101):
        g, j = mapping[aid]
        P_full[aid-1] = digit_arrays[g][j]
    return P_full

def simulate_live(ax, P_seq, t_seq, meta, fps=DEFAULT_FPS, dot=DEFAULT_DOT_SIZE, title=""):
    """Live 2D animation (XY projection) with 3D kinematics and reactive repulsion."""
    N = 100
    dt = 1.0/fps
    scale_xy = 100.0
    z0 = meta["z_m"]
    # Simple role-based z-layer for extra safety
    z_layer = np.zeros(N)
    # Random small jitter to avoid singular start
    rng = np.random.default_rng(7)

    # Initialize at first target
    P0 = P_seq[0]
    X = np.column_stack([P0[:,0]*scale_xy + (rng.random(N)-0.5)*0.5,
                         P0[:,1]*scale_xy + (rng.random(N)-0.5)*0.5,
                         np.ones(N)*z0])
    V = np.zeros((N,3))
    lam = LAM0

    scat = ax.scatter(X[:,0]/scale_xy, X[:,1]/scale_xy, s=dot)
    ax.set_xlim(-0.2, max(1.0, np.max(P_seq[0][:,0]) + (len(P_seq)>1)*0.5))
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_title(title, fontsize=11)

    t_global = 0.0
    for seg_idx, (t_start, t_end) in enumerate(t_seq):
        frames = max(1, int(round((t_end - t_start)*fps)))
        T = np.column_stack([P_seq[seg_idx][:,0]*scale_xy, P_seq[seg_idx][:,1]*scale_xy, np.ones(N)*z0])
        for f in range(frames):
            # Plan (PD)
            A_plan = K_P*(T - X) - K_V*V
            # React (repulsion in XY)
            A_react = np.zeros_like(A_plan)
            for i in range(N):
                pi = X[i]
                for j in range(i+1, N):
                    pj = X[j]
                    dvec = pi - pj
                    d = np.linalg.norm(dvec[:2])
                    if d < R_ACT and d > 1e-6:
                        dir_ = dvec[:2]/d
                        mag  = K_REP * max(0.0, (1.0/d - 1.0/R_ACT))
                        axy  = mag * dir_
                        A_react[i,:2] += axy
                        A_react[j,:2] -= axy
            # Adaptive lambda (raise when near violation)
            near = False
            for i in range(0, N, 2):
                for j in range(i+1, N, 3):
                    if np.linalg.norm((X[i]-X[j])[:2]) < (R_SAFE + 0.5):
                        near = True; break
                if near: break
            lam = min(LAM_MAX, lam + LAM_UP) if near else max(LAM_MIN, lam - LAM_DN)

            # Blend & integrate
            A_cmd = lam*A_react + (1.0 - lam)*A_plan
            # cap accel
            norms = np.linalg.norm(A_cmd, axis=1)
            clip  = norms > A_MAX
            if np.any(clip):
                A_cmd[clip] = (A_cmd[clip].T * (A_MAX / norms[clip])).T
            V = V + A_cmd*dt
            # cap velocity
            speed = np.linalg.norm(V, axis=1)
            clipv = speed > V_MAX
            if np.any(clipv):
                V[clipv] = (V[clipv].T * (V_MAX / speed[clipv])).T
            X = X + V*dt

            # Update plot (note: normalize back to [0,~] for display)
            scat.set_offsets(np.column_stack([X[:,0]/scale_xy, X[:,1]/scale_xy]))
            plt.pause(dt)
            t_global += dt

def main(argv):
    try:
        agents, digits, meta = load_formations()
    except FileNotFoundError:
        print("[ERROR] 'digit_formations_100.json' tidak ditemukan.")
        sys.exit(1)

    # 1) Input tahun
    raw_year = input("Masukkan tahunnya (contoh: 2025 / 3020 / 4180): ").strip()
    try:
        seq = parse_year(raw_year)
    except ValueError as e:
        print("[ERROR]", e); sys.exit(2)

    # 2) Pilih mode
    raw_mode = input("Pilih mode ('seq' untuk berurutan/morph, 'multi' untuk tampil bersamaan) [default seq]: ").strip().lower()
    mode = raw_mode if raw_mode in ("seq","multi") else DEFAULT_MODE

    # 3) Timing & FPS (opsional)
    if mode == "seq":
        s = input(f"Durasi diam per digit (detik) [default {DEFAULT_DWELL}]: ").strip()
        dwell = float(s) if s else DEFAULT_DWELL
        s = input(f"Durasi morph antar digit (detik) [default {DEFAULT_MORPH}]: ").strip()
        morph = float(s) if s else DEFAULT_MORPH
        s = input(f"FPS [default {DEFAULT_FPS}]: ").strip()
        fps = int(s) if s else DEFAULT_FPS
        # Build targets
        P_list, t_list = build_targets_seq(digits, seq, dwell, morph)
        title = f"Tahun: {''.join(map(str, seq))} | Sequential"
        plt.figure(figsize=(6,5))
        simulate_live(plt.gca(), P_list, t_list, meta, fps=fps, dot=DEFAULT_DOT_SIZE, title=title)
        print("[INFO] Tutup jendela untuk mengakhiri.")
        plt.show()
    else:
        s = input(f"Durasi tampil bersamaan (detik) [default {DEFAULT_HOLD}]: ").strip()
        hold = float(s) if s else DEFAULT_HOLD
        s = input(f"FPS [default {DEFAULT_FPS}]: ").strip()
        fps = int(s) if s else DEFAULT_FPS
        P_full = build_targets_multi(digits, seq, spacing=DEFAULT_SPACING)
        # Simulate as a single segment (hold seconds)
        title = f"Tahun: {''.join(map(str, seq))} | Simultan"
        plt.figure(figsize=(7,5))
        simulate_live(plt.gca(), [P_full], [(0.0, hold)], meta, fps=fps, dot=DEFAULT_DOT_SIZE, title=title)
        print("[INFO] Tutup jendela untuk mengakhiri.")
        plt.show()

if __name__ == "__main__":
    main(sys.argv)
