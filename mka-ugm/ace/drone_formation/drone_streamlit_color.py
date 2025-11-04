import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
import io, tempfile

st.set_page_config(page_title="3D Agentic Drone Light Show", layout="wide")

st.title("🌠 3D Agentic Drone Light Show — Orbit + Trails Edition")

# --- User Inputs ---
word = st.text_input("Enter word (English or Indonesian):", "LOVE")
color1 = st.color_picker("Start Color (Color A):", "#00FFFF")  # cyan
color2 = st.color_picker("End Color (Color B):", "#FF69B4")    # pink
trail_length = st.slider("Drone trail length:", 5, 30, 15)
run_animation = st.button("Run 3D Drone Show")

# --- Parameters ---
N = 1000  # drones
width, height, depth = 18, 10, 8

# Initialize positions (N x 3)
positions = np.random.rand(N, 3) * [width, height, depth]

# --- Convert Text to Target Points ---
def text_to_points(text, font_size=150):
    font = ImageFont.truetype("arial.ttf", font_size)
    img = Image.new('L', (800, 200))
    d = ImageDraw.Draw(img)
    d.text((10, 10), text, font=font, fill=255)
    img = img.resize((180, 100))
    y, x = np.where(np.array(img) > 128)
    points = np.stack([x / 10, (100 - y) / 10], axis=1)
    return points

points_2d = text_to_points(word)
indices = np.random.choice(len(points_2d), N, replace=True)
targets_2d = points_2d[indices]

# Add random z-offset for depth
z_offsets = np.random.uniform(-1, 1, N)
targets = np.column_stack((targets_2d, z_offsets + depth / 2))

# --- Drone Utility Functions ---
def update_positions(positions, targets, step=0.05):
    diff = targets - positions
    positions += diff * step
    return positions

def interpolate_color(c1, c2, factor):
    c1 = np.array([int(c1[i:i+2], 16) for i in (1,3,5)])
    c2 = np.array([int(c2[i:i+2], 16) for i in (1,3,5)])
    rgb = (1 - factor) * c1 + factor * c2
    return np.clip(rgb / 255.0, 0, 1)

# --- 3D Figure Setup ---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

colors = np.tile(interpolate_color(color1, color2, 0.5), (N, 1))
scat = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=8, c=colors, depthshade=True)
trails = [ax.plot([], [], [], lw=0.5, color=colors[i])[0] for i in range(N)]

ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_zlim(0, depth)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"3D Drone Light Show: '{word}'")

# Store past positions for trails
past_positions = [np.zeros((trail_length, 3)) for _ in range(N)]

# --- Animation Function ---
def animate(frame):
    global positions, past_positions

    # Move drones
    positions = update_positions(positions, targets, step=0.05)

    # Save position history (trail)
    for i in range(N):
        past_positions[i] = np.roll(past_positions[i], -1, axis=0)
        past_positions[i][-1] = positions[i]

    # Pulse color
    pulse = (np.sin(frame / 10) + 1) / 2
    color_mix = interpolate_color(color1, color2, pulse)
    scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    scat.set_color(np.tile(color_mix, (N, 1)))

    # Update trails (light tails)
    for i, line in enumerate(trails):
        trail_data = past_positions[i]
        alpha = np.linspace(0, 1, trail_length)
        fade_color = np.tile(color_mix, (trail_length, 1)) * alpha[:, None]
        line.set_data(trail_data[:, 0], trail_data[:, 1])
        line.set_3d_properties(trail_data[:, 2])
        line.set_color(color_mix * 0.7)  # faint glow

    # Auto-rotate camera
    ax.view_init(elev=20, azim=(frame * 1.2) % 360)

    return scat, *trails

# --- Streamlit Rendering ---
if run_animation:
    st.write("🛫 Drones are forming the word... please wait")

    ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)

    # ✅ Save animation to a temporary GIF file
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        ani.save(tmpfile.name, writer='pillow', fps=30)
        tmpfile.seek(0)
        gif_bytes = tmpfile.read()

    st.image(gif_bytes, caption=f"Drone Formation Animation: '{word}'", use_container_width=True)
else:
    st.pyplot(fig)
