# tutorials from the first few videos of a playlist here:
# https://www.youtube.com/watch?v=R4n4NyDG2hI&t=37s
# this was from 8 years ago...going to just move to something else at this point
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

verticies = (
    (1,-1,-1),
    (1,1,-1),
    (-1, 1, -1),
    (-1,-1,-1),
    (1, -1, 1),
    (1,1,1),
    (-1, -1, 1),
    (-1, 1,1),
)

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7),
)

surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6),
)

colors = (
    (0,0,0),
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,1,0),
    (0,1,1),
    (1,0,1),
    (0,1,0),
    (0,0,1),
    (1,1,0),
    (0,1,1),
    (1,0,1),
)


def Cube():
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0

        for vertex in surface:
            x+=1
            glColor3fv(colors[x])
            glVertex3fv(verticies[vertex])

    glEnd()    


    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()
# 

def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1 ,50.0)
    glTranslatef(0,0,-15)
    glRotatef(0,0,0,0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
         
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    glTranslatef(-1,0,0) 
        # glRotatef(1,3,0,1)
                if event.key == pygame.K_RIGHT:
                    glTranslatef(1,0,0) 

                if event.key == pygame.K_UP:
                    glTranslatef(0,1,0) 

                if event.key == pygame.K_DOWN:
                    glTranslatef(0,-1,0) 


                if event.key == pygame.K_PAGEUP:
                    glTranslatef(0,0,-1) 

                if event.key == pygame.K_PAGEDOWN:
                    glTranslatef(0,0,1) 

                if event.key == pygame.K_j:
                    glRotatef(1,1,0,0)

                if event.key == pygame.K_k:
                    glRotatef(1,-1,0,0)

                if event.key == pygame.K_h:
                    glRotatef(1,0,1,0)

                if event.key == pygame.K_l:
                    glRotatef(1,0,-1,0)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(10)

main()