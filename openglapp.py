import glfw
import inspect
import time
from dataclasses import make_dataclass
from OpenGL.GL import * # type:ignore
from canvas import Canvas

class App:
    _callersGlobals = dict()
    width: int
    height: int
    mouseX: int = 0
    mouseY: int = 0
    canvas: Canvas

    def __init__(self, width=300, height=300):
        if not glfw.init():
            print("Init failed")

        self._callersGlobals = inspect.stack()[1][0].f_globals

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        self.window = glfw.create_window(width, height, "Hello, world", None, None)
        glfw.make_context_current(self.window)

        self.width, self.height = width, height

        self.timerDelay = 200

        self.lastTimer = time.time()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glFrontFace(GL_CW)
        glEnable(GL_BLEND)

        self.canvas = Canvas(self.width, self.height)

        self.run()
    
    def _callFn(self, fn, *args):
        if (fn in self._callersGlobals): self._callersGlobals[fn](*args)
    
    def run(self):
        self._callFn("appStarted", self)

        def sizeChanged(window, width, height):
            glViewport(0, 0, width, height)
            self._callFn('sizeChanged', self)
        
        def keyEvent(*args): self.dispatchKeyEvent(*args)
        def mouseMoved(*args): self.dispatchMouseMoved(*args)
        def mouseChanged(*args): self.dispatchMouseChanged(*args)
        
        glfw.set_framebuffer_size_callback(self.window, sizeChanged)
        glfw.set_key_callback(self.window, keyEvent)
        glfw.set_cursor_pos_callback(self.window, mouseMoved)
        glfw.set_mouse_button_callback(self.window, mouseChanged)

        i = 0

        times = [0.0] * 20

        while not glfw.window_should_close(self.window):
            times.pop(0)

            glClearColor(0.2, 0.3, 0.3, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) #type:ignore

            self._callFn('redrawAll', self, self.window, self.canvas)

            self.canvas.redraw()

            if time.time() - self.lastTimer > (self.timerDelay / 1000.0):
                self.lastTimer = time.time()
                self._callFn('timerFired', self)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

            times.append(time.time())
            
            i += 1

            if i % 50 == 0:
                timeDiff = (times[-1] - times[0]) / (len(times) - 1)
                print(timeDiff)

        self._callFn('appStopped', self)
        
        glfw.terminate()

    
    def dispatchKeyEvent(self, window, key: int, scancode: int, action: int, mods: int):
        if ((ord('A') <= key and key <= ord('Z'))
            or (ord('1') <= key and key <= ord('9'))):
            keyName = chr(key)
        elif key == glfw.KEY_ESCAPE:
            keyName = 'Escape'
        elif key == glfw.KEY_SPACE:
            keyName = 'Space'
        elif key == glfw.KEY_BACKSPACE:
            keyName = 'Backspace'
        else:
            keyName = None
        
        KeyEvent = make_dataclass('KeyEvent', ['window', 'key'])

        event = KeyEvent(window, keyName)
        
        if action != glfw.REPEAT:
            if keyName is None:
                print(f"Unknown key {key}")
            elif action == glfw.PRESS:
                self._callFn('keyPressed', self, event)
            else:
                self._callFn('keyReleased', self, event)
    
    def dispatchMouseMoved(self, window, mouseX, mouseY):
        self.mouseX = mouseX
        self.mouseY = mouseY

        MoveEvent = make_dataclass('MoveEvent', ['x', 'y'])
        self._callFn('mouseMoved', self, MoveEvent(mouseX, mouseY))
    
    def dispatchMouseChanged(self, window, button: int, action: int, mods: int):
        ChangeEvent = make_dataclass('ChangeEvent', ['x', 'y'])
        
        event = ChangeEvent(self.mouseX, self.mouseY)

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self._callFn('mousePressed', self, event)
            else:
                self._callFn('mouseReleased', self, event)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self._callFn('rightMousePressed', self, event)
            else:
                self._callFn('rightMouseReleased', self, event)

runApp = App