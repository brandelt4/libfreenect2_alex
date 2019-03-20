import tkinter


class Window():

    def __init__(self, message):
        self.mainWindow = tkinter.Tk()
        self.v = tkinter.StringVar()
        self.v.set(message)
        self.label = tkinter.Label(self.mainWindow, text='Current Activity: ', font=("Helvetica", 60))
        self.label.pack()
        self.label2 = tkinter.Label(self.mainWindow, textvariable=self.v, font=("Helvetica", 50), fg='green')
        self.label2.pack()

        button = tkinter.Button(self.mainWindow, text='Continue...', command=self.mainWindow.destroy).pack()

        self.mainWindow.mainloop()


    def close(self):
        self.mainWindow.destroy()


    def changeActivity(self, mess):
        global v
        v.set(mess)





# windowCreated = False
# try: mainWindow
# except NameError: mainWindow = None
# if mainWindow is None:
#     mainWindow = tkinter.Tk()
#     windowCreated = True


def changeActivity(mess):
    w = Window(mess)


w = Window(' ')