import json, numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

JSON_PATH = Path(__file__).with_name("digit_formations_100.json")

# === Load digit formations ===
def load_formations():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    agents = {a["agent_id"]: a for a in data["agents"]}
    digits = {int(k): sorted(v, key=lambda e: e["agent_id"]) for k, v in data["digits"].items()}
    meta = data["meta"]
    return agents, digits, meta

# === Helpers ===
def resample_digit_points(digits, d: int, k: int):
    base = digits[d]
    P = np.array([[e["x_norm"], e["y_norm"]] for e in base], dtype=float)
    if k >= len(base):
        return P.copy()
    idx = np.linspace(0, len(base) - 1, num=k, dtype=int)
    return P[idx]

def allocate_agents_for_multi(N_digits: int):
    base = 100 // N_digits
    rem = 100 % N_digits
    alloc = [base] * N_digits
    for i in range(rem):
        alloc[i] += 1
    mapping = {}
    aid = 1
    for g in range(N_digits):
        for j in range(alloc[g]):
            mapping[aid] = (g, j)
            aid += 1
    return alloc, mapping

def build_multi_digit_points(digits_list, digits_data, spacing=1.25):
    """Return array (100,2) of full multi-digit formation"""
    alloc, mapping = allocate_agents_for_multi(len(digits_list))
    digit_points = []
    for g, d in enumerate(digits_list):
        k = alloc[g]
        P = resample_digit_points(digits_data, d, k)
        x_offset = g * spacing
        P_off = P + np.array([x_offset, 0.0])
        digit_points.append(P_off)

    P_full = np.zeros((100, 2))
    for aid in range(1, 101):
        g, j = mapping[aid]
        P_full[aid - 1] = digit_points[g][j]
    return P_full

# === Visualize morphing between multi-digit formations ===
def visualize_multi_sequence(sequence_list, dwell=2.0, morph=1.5, fps=25, spacing=1.25):
    """
    sequence_list: list of multi-digit arrays, e.g. [[2,0,2,5],[3,2,4,1],[9,8,5,2]]
    """
    agents, digits, meta = load_formations()
    T_frame = 1.0 / fps
    frames = []

    # Build all target formations
    formations = [build_multi_digit_points(seq, digits, spacing) for seq in sequence_list]
    P_curr = formations[0]

    # Dwell on first
    for _ in range(int(dwell * fps)):
        frames.append(P_curr.copy())

    # Morph + dwell for each next sequence
    for idx in range(1, len(formations)):
        P_next = formations[idx]
        steps = int(round(morph * fps))
        # Morph transition
        for k in range(1, steps + 1):
            a = k / steps
            Pk = (1 - a) * P_curr + a * P_next
            frames.append(Pk)
        # Dwell on next
        P_curr = P_next
        for _ in range(int(dwell * fps)):
            frames.append(P_curr.copy())

    # === Animate ===
    fig, ax = plt.subplots(figsize=(8, 4))
    scat = ax.scatter([], [], s=30, c='cyan')

    total_digits = len(sequence_list[0])
    ax.set_xlim(-0.5, spacing * total_digits)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.set_facecolor("black")
    ax.set_title("Drone Show - Multi Digit Morph Sequence", color='white')
    ax.tick_params(colors='white')

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    def update(frame):
        scat.set_offsets(frame)
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  init_func=init, blit=True,
                                  interval=T_frame * 1000, repeat=True)
    plt.show()

# === Example usage ===
if __name__ == "__main__":
    visualize_multi_sequence(
        sequence_list=[
            [2, 0, 2, 5],
            [3, 2, 4, 1],
            [9, 8, 5, 2],
            [3, 4, 1, 4]
        ],
        dwell=1.5, morph=1.0, fps=25
    )
