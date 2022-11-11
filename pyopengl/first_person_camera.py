import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr
from PIL import Image
import time

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
RETURN_ACTION_CONTINUE = 0
RETURN_ACTION_END = 1

def initialize_glfw():

    glfw.init()
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(
        GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, 
        GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE
    )
    glfw.window_hint(
        GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, 
        GLFW_CONSTANTS.GLFW_TRUE
    )
    glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER, GL_FALSE)

    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "My Game", None, None)
    glfw.make_context_current(window)
    glfw.set_input_mode(
        window, 
        GLFW_CONSTANTS.GLFW_CURSOR, 
        GLFW_CONSTANTS.GLFW_CURSOR_HIDDEN
    )

    return window

class Cube:

    def __init__(self, position, eulers):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

class Camera:

    def __init__(self, position):

        self.position = np.array(position, dtype = np.float32)
        self.yaw = 0
        self.pitch = 0
        self.update_vectors()
    
    def update_vectors(self):

        self.forwards = np.array(
            [
                np.cos(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(self.pitch)),
                np.sin(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(self.pitch)),
                np.sin(np.deg2rad(self.pitch))
            ]
        )

        globalUp = np.array([0,0,1], dtype = np.float32)
        self.right = np.cross(self.forwards, globalUp)
        self.up = np.cross(self.right, self.forwards)

class Scene:

    def __init__(self):

        self.cubes = [
            Cube(
                position = [6,0,0],
                eulers = [0,0,0]
            ),
            Cube(
                position = [9,0,0],
                eulers = [0,0,0]
            )
        ]

        self.camera = Camera(position = [0,0,2])
    
    def update(self, rate):

        for cube in self.cubes:
            cube.eulers[1] += 0.0 * rate
            if cube.eulers[1] > 360:
                cube.eulers[1] -= 360
    
    def move_camera(self, dir):
        adjustment = .1 #* self.frameTime / 16.7 
        if dir == "forward":
            dPos = adjustment * self.camera.forwards
        if dir == "backward":
            dPos = adjustment * -self.camera.forwards
        if dir == "right":
            dPos = adjustment * self.camera.right
        if dir == "left":
            dPos = adjustment * -self.camera.right

        dPos = np.array(dPos, dtype = np.float32)
        self.camera.position += dPos
         
    def spin_camera(self, dyaw, dpitch):

        self.camera.yaw += dyaw
        if self.camera.yaw > 360:
            self.camera.yaw -= 360
        elif self.camera.yaw < 0:
            self.camera.yaw += 360
        
        self.camera.pitch = min(
            89, max(-89, self.camera.pitch + dpitch)
        )

        self.camera.update_vectors()
    
class App:
    def __init__(self, window):
        
        self.window = window
        self.renderer = GrapitchcsEngine()
        self.scene = Scene()

        self.lastTime = glfw.get_time()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0

        self.walk_offset_lookup = {
            1: 0,
            2: 90,
            4: 180,
            8: 270,
        }
        
        self.mainLoop()

    def mainLoop(self):
        running = True
        while (running):
            #check events
            
            if glfw.window_should_close(self.window) \
                or glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:

                running = False
            
            self.handleKeys()
            self.handleMouse()
            
            glfw.poll_events()

            # this is just about rotating the cubes which I'm not interested in
            # self.scene.update(self.frameTime / 16.7)

            self.renderer.render(self.scene)

            #timing
            time.sleep(.005)
            self.calculateFramerate()
        self.quit()
    
    def handleKeys(self):

        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_W) == GLFW_CONSTANTS.GLFW_PRESS:
            self.scene.move_camera("forward")
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_A) == GLFW_CONSTANTS.GLFW_PRESS:
            self.scene.move_camera("left")
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_S) == GLFW_CONSTANTS.GLFW_PRESS:
            self.scene.move_camera("backward")
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_D) == GLFW_CONSTANTS.GLFW_PRESS:
            self.scene.move_camera("right")


        # if combo in self.walk_offset_lookup:
        if True:
            # directionModifier = self.walk_offset_lookup[combo]
            # dPos = [
            #     0.1 * self.frameTime / 16.7 * (np.cos(np.deg2rad(self.scene.camera.yaw))*(np.cos(np.deg2rad(self.scene.camera.pitch)))),
            #     0.1 * self.frameTime / 16.7 * np.sin(np.deg2rad(self.scene.camera.yaw)),
            #     0.1 * self.frameTime / 16.7 * (np.sin(np.deg2rad(self.scene.camera.pitch))*(np.cos(np.deg2rad(self.scene.camera.pitch))))]
            #     # 0]
            print(f"Yaw:{self.scene.camera.yaw}")
            print(f"Pitch:{self.scene.camera.pitch}")
            # print(f"Direction Modifier: {directionModifier}")
            # print(dPos)
            # print(.1 * self.frameTime / 16.7 * self.scene.camera.forwards)
            # self.scene.move_camera(dPos)
        
    def handleMouse(self):

        (x,y) = glfw.get_cursor_pos(self.window)
        rate = self.frameTime / 16.7
        yaw_increment = rate * ((SCREEN_WIDTH/2) - x)
        pitch_increment = rate * ((SCREEN_HEIGHT/2) - y)
        # if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_SPACE) == GLFW_CONSTANTS.GLFW_PRESS:
        self.scene.spin_camera(yaw_increment, pitch_increment)
        glfw.set_cursor_pos(self.window, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    
    def calculateFramerate(self):

        self.currentTime = glfw.get_time()
        delta = self.currentTime - self.lastTime
        if (delta >= 1):
            framerate = max(1, int(self.numFrames / delta))
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0/max(1,framerate))
        self.numFrames += 1

    def quit(self):
        self.renderer.quit()

class GrapitchcsEngine:


    def __init__(self):

        self.wood_texture = Material("pyopengl/gfx/marble.jpg")
        self.cube_mesh = Mesh("pyopengl/models/cube.obj")

        #initialise opengl
        glClearColor(0.1, 0.2, 0.2, 1)
        self.shader = self.createShader("pyopengl/shaders/vertex.txt", "pyopengl/shaders/fragment.txt")
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        glEnable(GL_DEPTH_TEST)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 640/480, 
            near = 0.1, far = 100, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader,"projection"),
            1, GL_FALSE, projection_transform
        )
        self.modelMatrixLocation = glGetUniformLocation(self.shader,"model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader,"view")

    def createShader(self, vertexFilepath, fragmentFilepath):

        with open(vertexFilepath,'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath,'r') as f:
            fragment_src = f.readlines()
        
        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))
        
        return shader
    
    def render(self, scene):

        #refresh screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)

        view_transform = pyrr.matrix44.create_look_at(
            eye = scene.camera.position,
            target = scene.camera.position + scene.camera.forwards,
            up = scene.camera.up, dtype=np.float32)

        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)

        self.wood_texture.use()
        glBindVertexArray(self.cube_mesh.vao)
        for cube in scene.cubes:
            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform, 
                m2=pyrr.matrix44.create_from_eulers(
                    eulers=np.radians(cube.eulers), dtype=np.float32
                )
            )
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform, 
                m2=pyrr.matrix44.create_from_translation(
                    vec=np.array(cube.position),dtype=np.float32
                )
            )
            glUniformMatrix4fv(self.modelMatrixLocation,1,GL_FALSE,model_transform)
            glDrawArrays(GL_TRIANGLES, 0, self.cube_mesh.vertex_count)

        glFlush()
    
    def quit(self):
        self.cube_mesh.destroy()
        self.wood_texture.destroy()
        glDeleteProgram(self.shader)
        
class Mesh:
    def __init__(self, filename):
        # x, y, z, s, t, nx, ny, nz
        self.vertices = self.loadMesh(filename)
        self.vertex_count = len(self.vertices)//8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
    
    def loadMesh(self, filename):

        #raw, unassembled data
        v = []
        vt = []
        vn = []
        
        #final, assembled and packed result
        vertices = []

        #open the obj file and read the data
        with open(filename,'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag=="v":
                    #vertex
                    line = line.replace("v ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag=="vt":
                    #texture coordinate
                    line = line.replace("vt ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vt.append(l)
                elif flag=="vn":
                    #normal
                    line = line.replace("vn ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vn.append(l)
                elif flag=="f":
                    #face, three or more vertices in v/vt/vn form
                    line = line.replace("f ","")
                    line = line.replace("\n","")
                    #get the individual vertices for each line
                    line = line.split(" ")
                    faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        #break out into [v,vt,vn],
                        #correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        faceTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        faceNormals.append(vn[normal])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    """
                        eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                    """
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i+1)
                        vertex_order.append(i+2)
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)
                line = f.readline()
        return vertices
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class Material:

    
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        with Image.open(filepath, mode = "r") as image:
            image_width,image_height = image.size
            image = image.convert("RGBA")
            img_data = bytes(image.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))


if __name__ == "__main__":
    window = initialize_glfw()
    myApp = App(window)