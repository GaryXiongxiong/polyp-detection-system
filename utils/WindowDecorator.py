import PySimpleGUI as sg
import platform

class Window(sg.Window):

    def disable(self):
        if platform.system() == "Windows":
            super(Window, self).disable()
        else:
            self.hide()

    def enable(self):
        if platform.system() == "Windows":
            super(Window, self).enable()
        else:
            self.un_hide()
