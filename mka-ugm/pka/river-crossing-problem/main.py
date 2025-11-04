import pygame
import time

# --- LOGIKA SEARCH (DFS) ---

initial_state = ('L', 'L', 'L', 'L')
goal_state = ('R', 'R', 'R', 'R')

def is_valid(state):
    farmer, chicken, corn, fox = state
    if farmer != chicken and chicken == corn:  # ayam makan jagung
        return False
    if farmer != chicken and chicken == fox:   # rubah makan ayam
        return False
    return True

def get_successors(state):
    successors = []
    farmer, chicken, corn, fox = state
    items = ['farmer', 'chicken', 'corn', 'fox']
    candidates = [('farmer',), ('chicken',), ('corn',), ('fox',)]
    
    for move in candidates:
        new_state = list(state)
        # pindahkan petani
        new_state[0] = 'R' if farmer == 'L' else 'L'
        # pindahkan item tambahan (jika ada)
        if move[0] != 'farmer':
            idx = items.index(move[0])
            if state[idx] == farmer:  # hanya bisa dipindah kalau di sisi sama
                new_state[idx] = new_state[0]
            else:
                continue
        new_state = tuple(new_state)
        if is_valid(new_state):
            successors.append(new_state)
    return successors

def dfs():
    stack = [(initial_state, [initial_state])]
    visited = set()
    while stack:
        state, path = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        if state == goal_state:
            return path
        for succ in get_successors(state):
            stack.append((succ, path + [succ]))
    return None

# --- VISUALISASI PYGAME ---

pygame.init()
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Petani - Ayam - Jagung - Rubah")

FONT = pygame.font.SysFont("Arial", 20)

# warna
WHITE = (255,255,255)
BLACK = (0,0,0)
BROWN = (139,69,19)      # Farmer
RED = (200,50,50)        # Chicken
YELLOW = (200,200,50)    # Corn
ORANGE = (255,140,0)     # Fox

# posisi dasar
positions = {
    'L': 150,
    'R': 650
}

def lerp(a, b, t):
    """Linear interpolation between a and b with t in [0,1]."""
    return a + (b - a) * t

def animate_state(prev_state, next_state, duration=1.0, fps=60):
    """Animates the transition from prev_state to next_state."""
    frames = int(duration * fps)
    for frame in range(frames):
        t = frame / frames
        screen.fill((100,150,255))
        pygame.draw.rect(screen, (0,100,200), (WIDTH//2 - 50, 0, 100, HEIGHT))
        text = FONT.render(f"State: {next_state}", True, BLACK)
        screen.blit(text, (10, 10))

        for idx, (name, color, radius, y, label_color) in enumerate([
            ("Petani", BROWN, 25, 100, WHITE),      # Farmer
            ("Ayam", RED, 20, 180, WHITE),          # Chicken
            ("Jagung", YELLOW, 20, 260, BLACK),     # Corn
            ("Rubah", ORANGE, 25, 340, BLACK)       # Fox
        ]):
            prev_side = prev_state[idx]
            next_side = next_state[idx]
            x = lerp(positions[prev_side], positions[next_side], t)
            pygame.draw.circle(screen, color, (int(x), y), radius)
            label = FONT.render(name, True, label_color)
            screen.blit(label, (int(x) - label.get_width()//2, y - label.get_height()//2))
        pygame.display.flip()
        pygame.time.delay(int(1000 / fps))

def draw_state(state):
    screen.fill((100,150,255))  # warna background biru langit
    pygame.draw.rect(screen, (0,100,200), (WIDTH//2 - 50, 0, 100, HEIGHT))
    text = FONT.render(f"State: {state}", True, BLACK)
    screen.blit(text, (10, 10))
    farmer, chicken, corn, fox = state
    # Farmer
    pygame.draw.circle(screen, BROWN, (positions[farmer], 100), 25)
    farmer_text = FONT.render("Petani", True, WHITE)
    screen.blit(farmer_text, (positions[farmer] - farmer_text.get_width()//2, 100 - farmer_text.get_height()//2))
    # Chicken
    pygame.draw.circle(screen, RED, (positions[chicken], 180), 20)
    chicken_text = FONT.render("Ayam", True, WHITE)
    screen.blit(chicken_text, (positions[chicken] - chicken_text.get_width()//2, 180 - chicken_text.get_height()//2))
    # Corn
    pygame.draw.circle(screen, YELLOW, (positions[corn], 260), 20)
    corn_text = FONT.render("Jagung", True, BLACK)
    screen.blit(corn_text, (positions[corn] - corn_text.get_width()//2, 260 - corn_text.get_height()//2))
    # Fox
    pygame.draw.circle(screen, ORANGE, (positions[fox], 340), 25)
    fox_text = FONT.render("Rubah", True, BLACK)
    screen.blit(fox_text, (positions[fox] - fox_text.get_width()//2, 340 - fox_text.get_height()//2))
    pygame.display.flip()

def main():
    solution = dfs()
    running = True
    step = 0
    playing = False
    prev_state = solution[0]
    draw_state(prev_state)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not playing and step < len(solution) - 1:
                    playing = True

        if playing and step < len(solution) - 1:
            next_state = solution[step + 1]
            animate_state(prev_state, next_state, duration=1.0, fps=60)
            prev_state = next_state
            step += 1
            if step == len(solution) - 1:
                playing = False
        elif step == len(solution) - 1:
            draw_state(solution[step])
            text = FONT.render("Selesai! Tekan ESC untuk keluar.", True, BLACK)
            screen.blit(text, (WIDTH//2 - 120, HEIGHT - 40))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
        else:
            draw_state(solution[step])

        pygame.time.delay(10)

    pygame.quit()

if __name__ == "__main__":
    main()