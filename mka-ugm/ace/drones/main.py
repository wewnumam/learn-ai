import pygame
import random
import sys

# --- Settings ---
WIDTH, HEIGHT = 800, 400
TEXT = "HELLO WORLD"
FONT_SIZE = 120
DRONE_COUNT = 400
SPEED = 0.07

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Formation Optimized")
clock = pygame.time.Clock()

# --- Render text surface ---
font = pygame.font.SysFont("Arial", FONT_SIZE, bold=True)
text_surface = font.render(TEXT, True, (255, 255, 255))
text_rect = text_surface.get_rect(center=(WIDTH//2, HEIGHT//2))

# --- Extract target positions sparsely ---
targets = []
step = 4  # increase to skip more pixels for performance
for y in range(0, text_surface.get_height(), step):
    for x in range(0, text_surface.get_width(), step):
        if text_surface.get_at((x, y))[0] > 128:  # bright pixel
            targets.append((text_rect.left + x, text_rect.top + y))

# --- Downsample to match drone count ---
if len(targets) > DRONE_COUNT:
    targets = random.sample(targets, DRONE_COUNT)
else:
    DRONE_COUNT = len(targets)

# --- Drone class ---
class Drone:
    def __init__(self, start_pos, target):
        self.pos = pygame.Vector2(start_pos)
        self.target = pygame.Vector2(target)
        self.home = pygame.Vector2(target)
        self.scatter_target = pygame.Vector2(random.randint(0, WIDTH), random.randint(0, HEIGHT))
        self.color = (random.randint(200, 255), random.randint(180, 255), random.randint(180, 255))

    def update(self, forming):
        goal = self.home if forming else self.scatter_target
        self.pos += (goal - self.pos) * SPEED

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (int(self.pos.x), int(self.pos.y)), 3)

# --- Initialize drones ---
drones = [Drone((random.randint(0, WIDTH), random.randint(0, HEIGHT)), t) for t in targets]
forming = True  # drones currently forming text

# --- Main loop ---
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
            forming = not forming  # toggle scatter/formation

    # Update
    for d in drones:
        d.update(forming)

    # Draw
    screen.fill((10, 10, 20))
    for d in drones:
        d.draw(screen)

    # Info text
    info = pygame.font.SysFont(None, 24).render(
        "Press SPACE to scatter/reform", True, (200, 200, 200)
    )
    screen.blit(info, (10, 10))

    pygame.display.flip()
    clock.tick(60)
