import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.textpath import TextPath
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

# ===================== PARAMETER GLOBAL =====================
N_DRONES = 1000
CANVAS_W, CANVAS_H = 18.0, 10.0        # ukuran kanvas (unit "meter" virtual)
Z_LEVEL = 0.0                           # ketinggian konstan (bisa dibuat profil)
SAMPLE_STEP = 0.02                      # resolusi raster sampling untuk huruf
TARGET_RADIUS = 0.02                    # toleransi dekat kontur
DT = 0.03                               # timestep (s)
V_MAX = 2.0                             # m/s
A_MAX = 3.0                             # m/s^2
NEIGHBOR_R = 0.25                       # radius tetangga anti-tubruk
Kp, Kd, Kv = 3.0, 1.2, 2.0              # gains reaktif
LAMBDA = 0.65                           # λ : reaktif vs strategis
GAMMA = 0.9                             # γ : diskon utilitas
DWELL_STEPS = 45                        # durasi bertahan pada formasi
TRANSITION_STEPS = 220                  # langkah animasi transisi antar formasi

np.random.seed(7)

# ===================== UTIL: TEKS -> TITIK =====================
def text_to_points(word, n_points, canvas_w=CANVAS_W, canvas_h=CANVAS_H,
                   sample_step=SAMPLE_STEP, target_radius=TARGET_RADIUS):
    """
    Konversi word -> N titik (x,y,z) tersebar merata pada kontur/isi glyph.
    """
    # Buat TextPath (skala awal 1.0)
    tp = TextPath((0,0), word, size=1.0, prop=dict(family='DejaVu Sans', weight='bold'))
    bbox = tp.get_extents()  # in glyph coords
    gw, gh = bbox.width, bbox.height
    if gw == 0 or gh == 0:
        raise ValueError(f"Word '{word}' menghasilkan bounding box kosong.")

    # Skala agar muat ke canvas, beri margin
    scale = 0.9 * min(canvas_w/gw, canvas_h/gh)
    T = Affine2D().scale(scale).translate((canvas_w - gw*scale)/2 - bbox.x0*scale,
                                          (canvas_h - gh*scale)/2 - bbox.y0*scale)
    tp = T.transform_path(tp)
    poly = tp.vertices
    codes = tp.codes
    text_path = Path(poly, codes)

    # Raster grid
    xs = np.arange(0, canvas_w, sample_step)
    ys = np.arange(0, canvas_h, sample_step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Pilih titik yang di dalam atau dekat kontur
    inside = text_path.contains_points(pts, radius=target_radius)
    pts_in = pts[inside]

    if len(pts_in) < n_points:
        # Jika kurang padat, ambil titik terdekat dari kontur
        # (fallback sederhana: pakai semua yang ada lalu akan di-resample)
        base = pts_in
    else:
        base = pts_in

    # Pilih tepat n_points: sampling merata (random tanpa pengulangan)
    if len(base) >= n_points:
        idx = np.linspace(0, len(base)-1, n_points).astype(int)
        chosen = base[idx]
    else:
        # Gandakan acak sampai cukup
        reps = int(np.ceil(n_points / len(base)))
        til = np.tile(base, (reps,1))[:n_points]
        chosen = til

    # Tambah jitter kecil agar tidak terlalu grid-like
    jitter = (np.random.rand(len(chosen), 2) - 0.5) * sample_step * 0.5
    chosen = chosen + jitter

    # Tambah dimensi z
    z = np.full((len(chosen), 1), Z_LEVEL)
    points3d = np.hstack([chosen, z])
    return points3d

# ===================== ALLOKASI DRONE -> TARGET =====================
def greedy_match(current_xy, target_xy):
    """
    Greedy nearest-unique matching: O(N log N) kira-kira (dengan trik urut).
    current_xy, target_xy: (N,2)
    return perm: indeks target untuk masing-masing drone
    """
    N = current_xy.shape[0]
    # Ambil subset kandidat via grid hashing sederhana agar cepat
    perm = -np.ones(N, dtype=int)
    used = np.zeros(N, dtype=bool)

    # Urutkan drone berdasarkan posisi x (heuristik)
    order = np.argsort(current_xy[:,0])
    for i in order:
        # cari target terdekat yg belum dipakai
        d2 = np.sum((target_xy - current_xy[i])**2, axis=1)
        idx_sorted = np.argsort(d2)
        for j in idx_sorted[:50]:  # batasi ke 50 kandidat terdekat
            if not used[j]:
                perm[i] = j
                used[j] = True
                break

    # Resolusi konflik residu (jika ada -1)
    missing = np.where(perm < 0)[0]
    if len(missing) > 0:
        free_t = np.where(~used)[0]
        for i, j in zip(missing, free_t):
            perm[i] = j
            used[j] = True
    return perm

# ===================== KONTROL HIBRIDA (r & u) =====================
def separation_force(i, pos, neighbor_r=NEIGHBOR_R):
    # repulsi sederhana
    delta = pos[i] - pos
    dist2 = np.sum(delta**2, axis=1)
    mask = (dist2 > 1e-6) & (dist2 < neighbor_r**2)
    vec = delta[mask]
    d2 = dist2[mask][:,None]
    if vec.shape[0] == 0:
        return np.zeros(3)
    rep = (vec / d2).sum(axis=0)
    return rep

def hybrid_accel(i, pos, vel, target, a_prev, alpha_i=1.0, psi_i=1.0):
    # r_i: gaya reaktif gabungan
    f_goal = Kp*(target - pos[i]) - Kd*vel[i]
    f_sep = separation_force(i, pos)
    v_norm = np.linalg.norm(vel[i]) + 1e-9
    over = max(0.0, v_norm - V_MAX)
    f_lim = -Kv * over * (vel[i] / v_norm)
    f_react = f_goal + f_sep + f_lim
    r_i = np.linalg.norm(f_react) * alpha_i

    # u_i: utilitas 1-langkah (aproksimasi)
    # kandidat aksi = f_react yang dinormalisasi menuju target + haluskan a_prev
    a_cand = f_react
    # Skalar energi/kelancaran
    energy = -np.dot(a_cand, a_cand)
    smooth = -np.dot(a_cand - a_prev, a_cand - a_prev)
    progress = -np.linalg.norm(target - (pos[i] + DT*vel[i]))
    u_i = energy + 0.5*smooth + 1.5*progress

    # gabungan sesuai persamaan
    # maximize: LAMBDA * r_i + (1-LAMBDA) * psi_i * u_i
    score_grad = LAMBDA * (f_react / (np.linalg.norm(f_react)+1e-9)) \
               + (1.0 - LAMBDA) * psi_i * (2*a_cand - 2*(a_cand - a_prev))
    a = a_cand + 0.15 * score_grad  # satu langkah "naik" gradien
    # klip percepatan
    n = np.linalg.norm(a)
    if n > A_MAX:
        a = a * (A_MAX / n)
    return a

# ===================== SIMULASI & ANIMASI =====================
def simulate(words):
    # 1) Bangun semua formasi
    formations = [text_to_points(w, N_DRONES) for w in words]
    targets_xy_list = [f[:,:2] for f in formations]

    # 2) Inisialisasi posisi/kecepatan drone
    pos = np.zeros((N_DRONES, 3))
    pos[:,0] = np.random.rand(N_DRONES) * CANVAS_W
    pos[:,1] = np.random.rand(N_DRONES) * CANVAS_H
    pos[:,2] = Z_LEVEL
    vel = np.zeros_like(pos)
    a_prev = np.zeros_like(pos)

    # 3) Bobot alpha & psi (contoh: tepi formasi diberi bobot lebih besar)
    alpha = np.ones(N_DRONES)
    psi = np.ones(N_DRONES)

    fig, ax = plt.subplots(figsize=(10,6))
    scat = ax.scatter(pos[:,0], pos[:,1], s=6, alpha=0.9)
    ax.set_xlim(0, CANVAS_W); ax.set_ylim(0, CANVAS_H)
    ax.set_aspect('equal', 'box')
    title = ax.set_title("Agentic 1000-Drone Word Formation")

    # urutan semua frame target
    sequence = []
    for k, tgt_xy in enumerate(targets_xy_list):
        # alokasi
        perm = greedy_match(pos[:,:2], tgt_xy)
        tgt = np.zeros_like(pos)
        tgt[:,:2] = tgt_xy[perm]
        tgt[:,2] = Z_LEVEL
        # transisi
        for _ in range(TRANSITION_STEPS):
            sequence.append((tgt.copy(), f"Formasi: {words[k]} (transisi)"))
        # dwell
        for _ in range(DWELL_STEPS):
            sequence.append((tgt.copy(), f"Formasi: {words[k]} (stabil)"))

    def update(frame_idx):
        nonlocal pos, vel, a_prev
        tgt3d, caption = sequence[frame_idx]
        # langkah kontrol semua drone
        acc = np.zeros_like(pos)
        for i in range(N_DRONES):
            acc[i] = hybrid_accel(i, pos, vel, tgt3d[i], a_prev[i], alpha[i], psi[i])
        # integrasi
        vel = vel + DT*acc
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        mask = (speed > V_MAX)
        vel[mask[:,0]] *= (V_MAX / (speed[mask[:,0]] + 1e-9))
        pos = pos + DT*vel
        a_prev = acc

        # render
        scat.set_offsets(pos[:,:2])
        title.set_text(caption)
        return scat, title

    anim = FuncAnimation(fig, update, frames=len(sequence), interval=30, blit=False)
    plt.show()

if __name__ == "__main__":
    try:
        user_in = input("Masukkan kata/kalimat (pisahkan dengan koma): ").strip()
        # contoh: Love, Kekasih, AI
        words = [w.strip() for w in user_in.split(",") if len(w.strip())>0]
        if len(words) == 0:
            words = ["Love", "Kekasih"]
        simulate(words)
    except Exception as e:
        print("Terjadi kesalahan:", e)
