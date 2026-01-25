import pygame
import sys
import os
import argparse

from agent import PongAgent
from env import PongEnv
from config import *

def run_visual(weights_path):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Pong')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 64)


    env = PongEnv()
    state = env.reset()


    agent = PongAgent(weights_file=weights_path)
    right_wins = 0
    total_games = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.act(state, stochastic=True)

        # Step Env
        state, reward, done, info = env.step(action)

        # Draw
        screen.fill(pygame.Color('black'))

        #paddles height 100 width 20
        left = pygame.Rect(PADDLE_LEFT_X , int(env.left_y - PADDLE_HALF_H), PADDLE_WIDTH , PADDLE_HEIGHT)
        right = pygame.Rect(PADDLE_RIGHT_X , int(env.right_y - PADDLE_HALF_H), PADDLE_WIDTH , PADDLE_HEIGHT)

        ball = pygame.Rect(int(env.cube_x),int(env.cube_y),CUBE_SIZE,CUBE_SIZE)

        pygame.draw.rect(screen, pygame.Color('white'), left)
        pygame.draw.rect(screen, pygame.Color('white'), right)
        pygame.draw.rect(screen, pygame.Color('white'), ball)

        #scores
        left_text = font.render(str(env.left_score), True, pygame.Color('white'))
        right_text = font.render(str(env.right_score), True, pygame.Color('white'))

        screen.blit(left_text, (WIDTH // 4 - left_text.get_width() // 2,20))
        screen.blit(right_text, (3*WIDTH // 4 - right_text.get_width() // 2,20))

        pygame.display.flip()
        clock.tick(60)

        if(done):
            total_games += 1
            if info["winner"] == "right":
                right_wins += 1

            print(f"Game {total_games} | Winner: {info['winner']} | Win Rate: {right_wins/total_games*100:.1f}%")
            state = env.reset()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    run_visual("pong_best.pth")