# from a set of tutorials here:
# https://www.youtube.com/watch?v=LCK1qdp_HhQ&list=PLn3eTxaOtL2PDnEVNwOgZFm5xYPr4dUoR


import pygame as pg
from OpenGL.GL import *

class App:
    def __init__(self):

        #initialize python
        pg.init()
        pg.display.set_mode((640, 480), pg.OPENGL|pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
        # initialize opengl
        glClearColor(0.1, 0.2, 0.2, 1) 
        self.mainLoop()

    def mainLoop(self):
        running = True

        while running:
            # check events
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
                
            #refesh screen
            glClear(GL_COLOR_BUFFER_BIT)
            pg.display.flip()

            # timeing
            self.clock.tick(60)
        self.quit()


    def quit(self):
        pg.quit()

if __name__ == "__main__":
    myApp = App()