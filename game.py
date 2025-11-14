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
speed =5
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

    # Logic
    # Collision
    # Walls
    if cube_y <= 0:
        cube_y = 0
        cube_vy *= -1
    elif cube_y + cube_size >= HEIGHT:
        cube_y = HEIGHT - cube_size
        cube_vy *= -1
    # Paddle
    if cube.colliderect(left_paddle) and cube_vx < 0:
        # Move the cube to the edge of the paddle to avoid it getting stuck
        cube.x=left_paddle.right
        cube_vx *= -1

    # Reset if left the screen
    if cube_x >= WIDTH or cube_x+cube_size <= 0:
        cube_x, cube_y, cube_vx, cube_vy = reset_cube()

    cube = pygame.Rect(int(cube_x), int(cube_y), cube_size, cube_size)

    # Enemy algorithm
    cube_center_y=cube_y + cube_size / 2
    left_center_y = left_paddle.centery

    if      cube_center_y < left_center_y:
        left_paddle.y -= paddle_speed
    elif    cube_center_y > left_center_y:
        left_paddle.y += paddle_speed

    if left_paddle.top < 0:
        left_paddle.top=0
    elif left_paddle.bottom > HEIGHT:
        left_paddle.bottom=HEIGHT


    # Draw scene
    screen.fill((30, 30, 30))  # dark background
    pygame.draw.rect(screen, (255, 255, 255), left_paddle)
    pygame.draw.rect(screen, (255, 255, 255), right_paddle)
    pygame.draw.rect(screen, (255, 255, 255), cube)

    pygame.display.flip()      # update the full display
    clock.tick(60)             # limit to 60 FPS

pygame.quit()
sys.exit()
