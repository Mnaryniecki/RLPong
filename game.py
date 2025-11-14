import pygame
import sys
import random
import math

pygame.init()

# Window
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My First Pygame Window")

clock = pygame.time.Clock()  # controls FPS

# Font & scores
font = pygame.font.Font(None, 64)  # None = default font, 64 = size
left_score = 0
right_score = 0

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

# Constant parameters
cube_size = 20
speed =8
paddle_speed=4

def reset_cube():
    x = WIDTH / 2 - cube_size / 2
    y = HEIGHT / 2 - cube_size / 2

    # random angle but avoid purely horizontal directions
    go_right = random.choice([True, False])
    if go_right:
        # from -pi/4 to pi/4 (towards the right)
        angle = random.uniform(-math.pi / 4, math.pi / 4)
    else:
        # from 3pi/4 to 5pi/4 (towards the left)
        angle = random.uniform(3 * math.pi / 4, 5 * math.pi / 4)
    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed

    return x, y, vx, vy

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

    # Enemy algorithm
    if abs(cube.centery-left_paddle.centery) > 5:
        if cube.centery-left_paddle.centery > 0:
            left_paddle.y += paddle_speed
        else:
            left_paddle.y -= paddle_speed

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

    # Human player
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        right_paddle.y -= paddle_speed
    if keys[pygame.K_DOWN]:
        right_paddle.y += paddle_speed

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

    pygame.display.flip()      # update the full display
    clock.tick(60)             # limit to 60 FPS

pygame.quit()
sys.exit()
