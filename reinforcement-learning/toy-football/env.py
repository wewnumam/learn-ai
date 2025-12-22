import pygame
import numpy as np
import time

# --- Constants ---
GRID_W, GRID_H = 10, 6
CELL_SIZE = 60
SCREEN_W, SCREEN_H = GRID_W * CELL_SIZE, GRID_H * CELL_SIZE
FPS = 10

# Colors
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)
RED = (220, 20, 60)   # Agent
YELLOW = (255, 215, 0) # Ball
BLUE = (0, 0, 255)    # Goal
BLACK = (0, 0, 0)

class FootballEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("1-Agent Football Toy Env")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        
        # Define Goal Position (Fixed at right center)
        self.goal_pos = np.array([GRID_W - 1, GRID_H // 2])
        
        self.reset()

    def reset(self):
        """Resets the environment to start state."""
        # Agent starts at random position on left half
        self.agent_pos = np.array([np.random.randint(0, 3), np.random.randint(0, GRID_H)])
        
        # Ball starts exactly where agent is (possession)
        self.ball_pos = self.agent_pos.copy()
        
        self.has_ball = True
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Returns (agent_x, agent_y, ball_x, ball_y, has_ball)."""
        return (*self.agent_pos, *self.ball_pos, int(self.has_ball))

    def step(self, action):
        """
        Actions:
        0: Move Up
        1: Move Down
        2: Move Left
        3: Move Right
        4: Shoot (Kick)
        """
        reward = -1  # Step penalty to encourage speed
        
        if self.done:
            return self._get_state(), 0, True

        # --- Movement Logic ---
        if action < 4:
            # Proposed move
            move = {
                0: np.array([0, -1]), # Up
                1: np.array([0, 1]),  # Down
                2: np.array([-1, 0]), # Left
                3: np.array([1, 0])   # Right
            }[action]
            
            new_agent_pos = self.agent_pos + move
            
            # Check Bounds
            new_agent_pos[0] = np.clip(new_agent_pos[0], 0, GRID_W - 1)
            new_agent_pos[1] = np.clip(new_agent_pos[1], 0, GRID_H - 1)
            
            self.agent_pos = new_agent_pos
            
            # Dribbling: If agent has ball, ball moves with agent
            if self.has_ball:
                self.ball_pos = self.agent_pos.copy()

        # --- Shooting Logic ---
        elif action == 4:
            if self.has_ball:
                self.has_ball = False # Ball is kicked
                
                # Simple physics: Ball travels towards goal
                dist_to_goal = np.linalg.norm(self.goal_pos - self.ball_pos)
                
                # Check if aligned with goal (y-axis) and facing goal (x-axis)
                if self.agent_pos[1] == self.goal_pos[1]:
                    self.ball_pos = self.goal_pos.copy()
                    reward = 100 # GOAL!
                    self.done = True
                else:
                    # Missed shot
                    self.ball_pos = np.array([GRID_W - 1, self.agent_pos[1]])
                    reward = -10 # Miss penalty
                    self.done = True # Game over on miss
            else:
                reward = -5 # Penalty for kicking air

        return self._get_state(), reward, self.done

    def render(self):
        self.screen.fill(GREEN)
        
        # Draw Grid
        for x in range(0, SCREEN_W, CELL_SIZE):
            pygame.draw.line(self.screen, WHITE, (x, 0), (x, SCREEN_H))
        for y in range(0, SCREEN_H, CELL_SIZE):
            pygame.draw.line(self.screen, WHITE, (0, y), (SCREEN_W, y))

        # Draw Goal
        goal_rect = pygame.Rect(self.goal_pos[0] * CELL_SIZE, self.goal_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, BLUE, goal_rect)
        
        # Draw Agent
        agent_center = (self.agent_pos[0] * CELL_SIZE + CELL_SIZE//2, self.agent_pos[1] * CELL_SIZE + CELL_SIZE//2)
        pygame.draw.circle(self.screen, RED, agent_center, CELL_SIZE//3)
        
        # Draw Ball
        ball_center = (self.ball_pos[0] * CELL_SIZE + CELL_SIZE//2, self.ball_pos[1] * CELL_SIZE + CELL_SIZE//2)
        # Offset ball slightly if held by agent to make it visible
        if self.has_ball:
            ball_center = (ball_center[0] + 10, ball_center[1] + 10)
            
        pygame.draw.circle(self.screen, YELLOW, ball_center, CELL_SIZE//5)

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()

# --- Manual Control Loop ---
if __name__ == "__main__":
    env = FootballEnv()
    running = True
    
    print("Controls: Arrow Keys to Move, SPACE to Shoot")
    
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action = 0
                elif event.key == pygame.K_DOWN: action = 1
                elif event.key == pygame.K_LEFT: action = 2
                elif event.key == pygame.K_RIGHT: action = 3
                elif event.key == pygame.K_SPACE: action = 4
        
        if action is not None:
            state, reward, done = env.step(action)
            print(f"Action: {action}, Reward: {reward}, State: {state}")
            if done:
                print("--- Episode Finished ---")
                time.sleep(1)
                env.reset()
        
        env.render()
    
    env.close()