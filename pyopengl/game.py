# from a set of tutorials here:
# https://www.youtube.com/watch?v=LCK1qdp_HhQ&list=PLn3eTxaOtL2PDnEVNwOgZFm5xYPr4dUoR


import pygame as pg
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader

class App:
    def __init__(self):

        #initialize python
        pg.init()
        pg.display.set_mode((640, 480), pg.OPENGL|pg.DOUBLEBUF)
        self.clock = pg.time.Clock()

        # initialize opengl
        glClearColor(0.1, 0.1, 0.1, 1)
        self.shader = self.createShader("pyopengl/shaders/vertex.txt", "pyopengl/shaders/fragment.txt") 
        glUseProgram(self.shader)
        self.triangle = Triangle()
        self.mainLoop()


    def createShader(self, vertexFilepath, fragmentFilepath):
        with open(vertexFilepath, 'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath, 'r') as f:
            fragment_src = f.readlines()
            print(fragment_src)

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))

        return shader

    def mainLoop(self):
        running = True

        while running:
            # check events
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
                
            #refesh screen
            glClear(GL_COLOR_BUFFER_BIT)

            glUseProgram(self.shader) # get shader ready. best practice since shaders may change
            glBindVertexArray(self.triangle.vao) # prepare thing to be drawn
            glDrawArrays(GL_TRIANGLES, 0, self.triangle.vertex_count)
            pg.display.flip()
            # timeing
            self.clock.tick(60)
        self.quit()


    def quit(self):
        self.triangle.destroy()
        glDeleteProgram(self.shader)
        pg.quit()


class Triangle:
    def __init__(self):
        
        # x,y,z,r,g,b
        # position is bare minimum property of a vertex
        self.vertices = (
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0, # each line is a vertex
             0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
             0.0,  0.5, 0.0, 0.0, 0.0, 1.0,
        )

        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vertex_count = 3

        self.vao = glGenVertexArrays(1) # vertex array objects are best way to contain vertex data
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1) # vao knows ot associate with vbo since it was declared earlier

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0) # position
        glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, 24 ,ctypes.c_void_p(0))
        glEnableVertexAttribArray(1) # color 
        glVertexAttribPointer(1,3, GL_FLOAT, GL_FALSE, 24 ,ctypes.c_void_p(12))  # 12 here indicates offsets in the vertex...to get to color get through 3 positional arguments of 4 bytes each

    def destroy(self):
        # release memory of GPU
        glDeleteVertexArrays(1,(self.vao,)) # detail...need to wrap variable to be freed in a list type
        glDeleteBuffers(1, (self.vbo,))


if __name__ == "__main__":
    myApp = App()