import pygame
import sys
import random
import math

from networkx.generators import stochastic
from sympy import elliptic_k
from sympy.physics.units import action

from agent import PongAgent

MAX_FRAMES_PER_GAME = 5000  # tune this

frames_this_game = 0

def teacher_action_from_state(state_check, margin=0.05):
    # state is the 8-element list from the game
    ball = state_check[1]    # cube_y / HEIGHT
    paddle = state_check[4]  # right_paddle.centery / HEIGHT
    diff = ball - paddle

    if diff < -margin:
        return 0  # up
    elif diff > margin:
        return 2  # down
    else:
        return 1  # stay

pygame.init()
agent = PongAgent()
# Window
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My First Pygame Window")

clock = pygame.time.Clock()  # controls FPS

# Font & scores
font = pygame.font.Font(None, 64)  # None = default font, 64 = size
left_score = 0
right_score = 0

GAMES_TO_PLAY = 50
POINTS_TO_WIN = 5

games_played = 0
right_wins = 0

# counter of the model following training remove later
ctr=0
decisions_made = 0

left_paddle = pygame.Rect(
    50,                 # x position
    HEIGHT // 2 - 50,   # y position (centered vertically)
    20,                 # width
    100                 # height
)


right_paddle = pygame.Rect(
    WIDTH - 50 - 30,    # x (50px from right edge)
    HEIGHT // 2 - 50,   # y
    20,                 # width
    100                  # height
)

left_paddle_dir = 0
right_paddle_dir = 0

# Constant parameters
cube_size = 20
speed =10
paddle_speed=4

def reset_cube():
    x = WIDTH / 2 - cube_size / 2
    y = HEIGHT / 2 - cube_size / 2

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

    return x, y, vx, vy


cube_x, cube_y, cube_vx, cube_vy = reset_cube()


def reset_game():
    global left_score, right_score
    global cube_x, cube_y, cube_vx, cube_vy
    global left_paddle, right_paddle
    global left_paddle_dir, right_paddle_dir
    global frames_this_game

    left_score = 0
    right_score = 0

    left_paddle.y = HEIGHT // 2 - 50
    right_paddle.y = HEIGHT // 2 - 50

    left_paddle_dir = 0
    right_paddle_dir = 0

    frames_this_game = 0

    cube_x, cube_y, cube_vx, cube_vy = reset_cube()


running = True
while running:
    # 1. Handle events (keyboard, mouse, quit)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update game state
    cube_x += cube_vx
    cube_y += cube_vy

    cube = pygame.Rect(int(cube_x), int(cube_y), cube_size, cube_size)

    # Logic
    # Collision
    # Walls
    if cube_y <= 0:
        cube_y = 0
        cube_vy *= -1
    elif cube_y + cube_size >= HEIGHT:
        cube_y = HEIGHT - cube_size
        cube_vy *= -1
    # Paddles
    if cube.colliderect(left_paddle) and cube_vx < 0:
        relative_intersect_y = (cube.centery -left_paddle.centery)/(left_paddle.height/2)
        #clipping to range <-1,1>
        relative_intersect_y = max(-1.0, min(relative_intersect_y , 1.0))
        angle = relative_intersect_y * ( 4 * math.pi ) / 12


        # Move the cube to the edge of the paddle to avoid it getting stuck
        cube.x=left_paddle.right
        cube_vx *= -1
        cube_vx = math.cos(angle) * speed
        cube_vy = math.sin(angle) * speed

    if cube.colliderect(right_paddle) and cube_vx > 0:
        relative_intersect_y = (cube.centery - right_paddle.centery)/(right_paddle.height/2)
        #clipping to range <-1,1>
        relative_intersect_y = max(-1.0, min(relative_intersect_y , 1.0))
        angle = math.pi - relative_intersect_y * ( 4 * math.pi ) / 12


        # Move the cube to the edge of the paddle to avoid it getting stuck
        cube.x=right_paddle.left
        cube_vx *= -1
        cube_vx = math.cos(angle) * speed
        cube_vy = math.sin(angle) * speed

    # Scoring & reset if cube leaves the screen
    if cube_x > WIDTH:
        # right side out → left player scores
        left_score += 1
        cube_x, cube_y, cube_vx, cube_vy = reset_cube()

    elif cube_x + cube_size < 0:
        # left side out → right player scores
        right_score += 1
        cube_x, cube_y, cube_vx, cube_vy = reset_cube()

    if frames_this_game >= MAX_FRAMES_PER_GAME:
        games_played += 1
        right_wins += 1

        print(f"Game {games_played} timed out Score L:{left_score} R:{right_score}")

        if games_played > GAMES_TO_PLAY:
            running = False
        else:
            reset_game()

    if left_score >= POINTS_TO_WIN or right_score >= POINTS_TO_WIN:
        games_played += 1
        if right_score > left_score:
            right_wins += 1


        print(f"Game: {games_played} finished. L:{left_score} R:{right_score}")
        print(f"Right wins so far: {right_wins}/{games_played}")
        print("-"*50)

        if games_played > GAMES_TO_PLAY:
            running = False
        else:
            reset_game()


    # Enemy algorithm
    if abs(cube.centery-left_paddle.centery) > 5:
        if cube.centery-left_paddle.centery > 0:
            left_paddle.y += paddle_speed
            left_paddle_dir = 1
        else:
            left_paddle.y -= paddle_speed
            left_paddle_dir = -1
    else:
        left_paddle_dir = 0

    if left_paddle.top < 0:
        left_paddle.top=0
    elif left_paddle.bottom > HEIGHT:
        left_paddle.bottom=HEIGHT

    '''
    # Right paddle temporary algorithm
    if abs(cube.centery-right_paddle.centery) > 5:
        if cube.centery-right_paddle.centery > 0:
            right_paddle.y += paddle_speed
        else:
            right_paddle.y -= paddle_speed

    '''
    '''
    # Human player
    keys = pygame.key.get_pressed()
    up=keys[pygame.K_UP]
    down=keys[pygame.K_DOWN]

    if (up and down) or (not up and not down):
        right_paddle_dir = 0
    else:
        if up:
            right_paddle_dir = -1
        elif down:
            right_paddle_dir = 1
    '''

    if right_paddle.top < 0:
        right_paddle.top=0
    elif right_paddle.bottom > HEIGHT:
        right_paddle.bottom=HEIGHT

    # Draw scene
    screen.fill((30, 30, 30))  # dark background
    pygame.draw.rect(screen, (255, 255, 255), left_paddle)
    pygame.draw.rect(screen, (255, 255, 255), right_paddle)
    pygame.draw.rect(screen, (255, 255, 255), cube)

    # --- Score rendering ---
    left_text = font.render(str(left_score), True, (255, 255, 255))
    right_text = font.render(str(right_score), True, (255, 255, 255))

    # position: left score at 1/4 width, right score at 3/4 width
    screen.blit(left_text, (WIDTH // 4 - left_text.get_width() // 2, 20))
    screen.blit(right_text, (3 * WIDTH // 4 - right_text.get_width() // 2, 20))

    # Saving the game state for the NN
    state = [
        cube_x / WIDTH,
        cube_y / HEIGHT,
        cube_vx / speed,  # roughly in [-1, 1]
        cube_vy / speed,  # roughly in [-1, 1]
        right_paddle.centery / HEIGHT,
        right_paddle_dir,
        left_paddle.centery / HEIGHT,
        left_paddle_dir
    ]

    action =agent.act(state , stochastic=False)
    teacher = teacher_action_from_state(state)

    #print("STATE:", state)
    print("TEACHER:", teacher, "NN:", action)
    '''
    if teacher == action:
        ctr += 1
    decisions_made += 1
    rate = ctr / decisions_made
    print("Success rate:" , rate)
    '''
    #print("-" * 40)
    # Action to direction
    if action == 0:
        right_paddle_dir = -1
    elif action == 1:
        right_paddle_dir = 0
    elif action == 2:
        right_paddle_dir = 1

    right_paddle.y += right_paddle_dir * paddle_speed

    frames_this_game += 1
    pygame.display.flip()      # update the full display
    #clock.tick(60)             # limit to 60 FPS

pygame.quit()
sys.exit()
