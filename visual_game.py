import pygame
import sys

from agent import PongAgent
from env import PongEnv, WIDTH, HEIGHT, cube_size

def run_visual():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Pong')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 64)


    env = PongEnv()
    state = env.reset()


    agent = PongAgent()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.act(state, stochastic=False)

        # Step Env
        state, reward, done, info = env.step(action)

        # Draw
        screen.fill(pygame.Color('black'))

        #paddles height 100 width 20
        left = pygame.Rect(50 , int(env.left_y-50),20 ,100)
        right = pygame.Rect(WIDTH-50-20 , int(env.right_y-50),20 ,100)

        ball = pygame.Rect(int(env.cube_x),int(env.cube_y),cube_size,cube_size)

        pygame.draw.rect(screen, pygame.Color('white'), left)
        pygame.draw.rect(screen, pygame.Color('white'), right)
        pygame.draw.rect(screen, pygame.Color('white'), ball)

        #scores
        left_text = font.render(str(env.left_score), True, pygame.Color('white'))
        right_text = font.render(str(env.right_score), True, pygame.Color('white'))

        screen.blit(left_text, (WIDTH // 4 - left_text.get_width() // 2,20))
        screen.blit(right_text, (3*WIDTH // 4 - right_text.get_width() // 2,20))

        pygame.display.flip()
        #clock.tick(60)

        if(done):
            state = env.reset()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    run_visual()