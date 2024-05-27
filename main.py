import threading
import time
from UserInterface import *
from AudioFiles import *

def update_windows():
    while True:
        print("test2")


print("test1")

app = MyApp()
threading.Thread(target=update_windows).start()

app.MainLoop()