import tkinter

windowCreated = False
try: mainWindow
except NameError: mainWindow = None
if mainWindow is None:
    mainWindow = tkinter.Tk()
    windowCreated = True

v = tkinter.StringVar()


class Window():

    def __init__(self):
        global label2
        v.set('nothing...')
        label = tkinter.Label(mainWindow, text='Current Activity: ')
        label.pack()
        label2 = tkinter.Label(mainWindow, textvariable=v)
        label2.pack()
        mainWindow.mainloop()


    def changeActivity(self, mess):
        global v
        v.set(mess)


if __name__ == '__main__':
    w = Window()