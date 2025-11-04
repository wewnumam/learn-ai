import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw, ImageFont
import io, tempfile

st.set_page_config(page_title="Agentic 1000-Drone Word Formation", layout="wide")

st.title("🚁 Agentic 1000-Drone Word Formation System")

# User input
word = st.text_input("Enter word (English or Indonesian):", "LOVE")
run_animation = st.button("Run Animation")

# Parameters
N = 1000  # number of drones
width, height = 18, 10

# Step 1: Generate random initial positions
positions = np.random.rand(N, 2) * [width, height]

# Step 2: Convert word to points
def text_to_points(text, font_size=150):
    font = ImageFont.truetype("arial.ttf", font_size)
    img = Image.new('L', (800, 200))
    d = ImageDraw.Draw(img)
    d.text((10, 10), text, font=font, fill=255)
    img = img.resize((180, 100))
    y, x = np.where(np.array(img) > 128)
    points = np.stack([x / 10, (100 - y) / 10], axis=1)
    return points

target_points = text_to_points(word)
indices = np.random.choice(len(target_points), N, replace=True)
targets = target_points[indices]

# Step 3: Move drones toward target
def update_positions(positions, targets, step=0.05):
    diff = targets - positions
    positions += diff * step
    return positions

# Step 4: Animation setup
fig, ax = plt.subplots(figsize=(10, 5))
scat = ax.scatter(positions[:, 0], positions[:, 1], s=5)
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_title(f"Agentic Drone Word Formation: '{word}'")

def animate(i):
    global positions
    positions = update_positions(positions, targets)
    scat.set_offsets(positions)
    return scat,

# Step 5: Streamlit animation display
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
