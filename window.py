import tkinter

# windowCreated = False
# try: mainWindow
# except NameError: mainWindow = None
# if mainWindow is None:
#     mainWindow = tkinter.Tk()
#     windowCreated = True



class Window():

    def __init__(self):
        mainWindow = tkinter.Tk()
        v = tkinter.StringVar()
        v.set('gkllkj')
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