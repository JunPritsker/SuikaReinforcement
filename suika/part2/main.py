import sys

import pygame
import pymunk

from cloud import Cloud
from collision import collide
from config import config, CollisionTypes
from particle import Particle
from text import score, gameover
from wall import Wall


screen = pygame.display.set_mode((config.screen.width, config.screen.height))
pygame.display.set_caption("PySuika")
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0, config.physics.gravity)
space.damping = config.physics.damping
space.collision_bias = config.physics.bias

left = Wall(config.top_left, config.bot_left, space)
bottom = Wall(config.bot_left, config.bot_right, space)
right = Wall(config.bot_right, config.top_right, space)
walls = [left, bottom, right]
cloud = Cloud()

wait_for_next = 0
game_over = False

handler = space.add_collision_handler(CollisionTypes.PARTICLE, CollisionTypes.PARTICLE)
handler.begin = collide
handler.data["score"] = 0

while not game_over:
    if pygame.event.peek():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and wait_for_next == 0:
                cloud.release(space)
                wait_for_next = config.screen.delay
            
    if wait_for_next > 1:
        wait_for_next -= 1
    if wait_for_next == 1:
        cloud.step()
        wait_for_next -= 1

    cloud.curr.set_x(pygame.mouse.get_pos()[0])

    screen.blit(config.background_blit, (0, 0))
    cloud.draw(screen, wait_for_next)

    for p in space.shapes:
        if not isinstance(p, Particle):
            continue
        p.draw(screen)
        # You lose the game when a fruit is above the kill line and is touching another fruit that's different from it
        # That means you can merge fruit above the kill line without losing as long as after the merge, the highest fruit is
        # below the kill line
        if p.pos[1] < config.pad.killy and p.has_collided:
            gameover(screen)
            game_over = True

    score(handler.data['score'], screen)

    space.step(1/config.screen.fps)
    pygame.display.update()
    clock.tick(config.screen.fps)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            pygame.quit()
            sys.exit()
