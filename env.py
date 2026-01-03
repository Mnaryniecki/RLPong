import random
import math

WIDTH, HEIGHT = 800, 600
cube_size = 20
speed =10
paddle_speed=4
POINTS_TO_WIN = 5
MAX_FRAMES_PER_GAME = 5000  # tune this

class PongEnv:
    def __init__(self):
        self.reset()

    def is_catchable(self, start_x, start_y, vx, vy):
        """
        Simulates the ball trajectory to check if the right paddle can physically reach it.
        """
        # If moving left, it's not the agent's problem (yet)
        if vx < 0:
            return True

        # Simulation variables
        sim_x, sim_y = start_x, start_y
        sim_vx, sim_vy = vx, vy
        
        # Target X is the front of the right paddle
        right_paddle_x = WIDTH - 50 - 20
        
        steps = 0
        # Simulate until ball passes paddle
        while sim_x + cube_size < right_paddle_x:
            sim_x += sim_vx
            sim_y += sim_vy

            # Wall bounces
            if sim_y <= 0:
                sim_y = 0
                sim_vy *= -1
            elif sim_y + cube_size >= HEIGHT:
                sim_y = HEIGHT - cube_size
                sim_vy *= -1
            
            steps += 1
            if steps > 1000: return True # Safety break

        # Ball center at impact
        ball_center_y = sim_y + cube_size / 2
        
        # How far can the paddle move in this time?
        max_travel_dist = paddle_speed * steps
        
        # Distance needed to cover (paddle center to ball center)
        # We subtract 50 because we only need the edge of the paddle to touch the ball center
        dist_needed = abs(ball_center_y - self.right_y) - 50
        
        # If we are already close enough, dist_needed is negative, so it's catchable
        return dist_needed <= max_travel_dist

    def reset_cube(self):
        x = WIDTH / 2 - cube_size / 2
        y = HEIGHT / 2 - cube_size / 2

        # Try to find a catchable angle (limit retries to avoid infinite loop)
        for _ in range(100):
            # random angle but avoid purely horizontal directions
            go_right = random.choice([True, False])
            if go_right:
                # from -pi/4 to pi/4 (towards the right)
                r_angle = random.uniform(-math.pi / 4, math.pi / 4)
            else:
                # from 3pi/4 to 5pi/4 (towards the left)
                r_angle = random.uniform(3 * math.pi / 4, 5 * math.pi / 4)
            
            vx = math.cos(r_angle) * speed
            vy = math.sin(r_angle) * speed
            
            # Check if this trajectory is fair
            if self.is_catchable(x, y, vx, vy):
                break
                
        return x, y, vx, vy

    def reset(self):
        self.left_score = 0
        self.right_score = 0

        self.left_y = HEIGHT // 2
        self.right_y = HEIGHT // 2
        self.left_dir =0
        self.right_dir = 0

        self.frames = 0

        self.cube_x, self.cube_y, self.cube_vx, self.cube_vy = self.reset_cube()
        self.done = False
        self.winner = None

        return self.get_state()

    def get_state(self):
        return [
        self.cube_x / WIDTH,
        self.cube_y / HEIGHT,
        self.cube_vx / speed,
        self.cube_vy / speed,
        self.right_y / HEIGHT,
        self.right_dir,
        self.left_y / HEIGHT,
        self.left_dir,
        ]

    def step(self, action):
        if self.done:
            return self.get_state(), 0.0, True, {}

        reward = 0.0

        # Action to direction
        if action == 0:
            self.right_dir = -1
        elif action == 1:
            self.right_dir = 0
        elif action == 2:
            self.right_dir = 1

        if abs(self.cube_y-self.left_y) > 5:
            if self.cube_y-self.left_y > 0:
                self.left_y += paddle_speed
                self.left_dir = 1
            else:
                self.left_y -= paddle_speed
                self.left_dir = -1
        else:
            self.left_dir = 0

        # Stopping the paddle from going offscreen
        half_h = 50 #half of the paddle
        self.left_y = max(half_h, min(self.left_y, HEIGHT - half_h))

        self.right_y += self.right_dir*paddle_speed
        self.right_y = max(half_h, min(self.right_y, HEIGHT - half_h))

        self.cube_x += self.cube_vx
        self.cube_y += self.cube_vy

        if self.cube_y <= 0:
            self.cube_y = 0
            self.cube_vy *= -1
        elif self.cube_y + cube_size >= HEIGHT:
            self.cube_y = HEIGHT - cube_size
            self.cube_vy *= -1


        #paddle x position
        left_x = 50
        right_x = WIDTH - 50 - 20

        if self.cube_x <= left_x and self.cube_vx < 0:
            if abs(self.cube_y + cube_size/2 -self.left_y) <= half_h:
                rel = (self.cube_y + cube_size/2 - self.left_y)/half_h
                #clipping to range <-1,1>
                rel = max(-1.0, min(rel , 1.0))
                angle = rel * ( 4 * math.pi ) / 12
                # Move the cube to the edge of the paddle to avoid it getting stuck
                self.cube_x= left_x+20
                self.cube_vx = math.cos(angle) * speed
                self.cube_vy = math.sin(angle) * speed

        if self.cube_x + cube_size >= right_x and self.cube_vx > 0:
            if abs(self.cube_y + cube_size / 2 - self.right_y) <= half_h:
                rel = (self.cube_y + cube_size / 2 - self.right_y) / half_h
                # clipping to range <-1,1>
                rel = max(-1.0, min(rel, 1.0))
                angle = math.pi - rel * (4 * math.pi) / 12

                # Move the cube to the edge of the paddle to avoid it getting stuck
                self.cube_x = right_x -cube_size
                self.cube_vx = math.cos(angle) * speed
                self.cube_vy = math.sin(angle) * speed
        # scoring when ball leaves the screen
        if self.cube_x > WIDTH:
            self.left_score += 1
            reward = -1.0
            self.cube_x, self.cube_y, self.cube_vx, self.cube_vy = self.reset_cube()

        elif self.cube_x + cube_size < 0:
            self.right_score += 1
            reward = +1.0
            self.cube_x, self.cube_y, self.cube_vx, self.cube_vy = self.reset_cube()

        self.frames += 1
        if self.frames >= MAX_FRAMES_PER_GAME:
            self.done = True
            self.winner = "timeout"
        elif self.left_score >= POINTS_TO_WIN or self.right_score >= POINTS_TO_WIN:
            self.done = True
            if self.right_score > self.left_score:
                self.winner = "right"
            elif self.left_score > self.right_score:
                self.winner = "left"
            else:
                self.winner = "none"
        info = {
            "winner": self.winner,
            "left_score": self.left_score,
            "right_score": self.right_score,
            "frames": self.frames,
        }
        return self.get_state(), reward, self.done, info