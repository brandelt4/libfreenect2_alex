import tkinter

# windowCreated = False
# try: mainWindow
# except NameError: mainWindow = None
# if mainWindow is None:
#     mainWindow = tkinter.Tk()
#     windowCreated = True



class Window():

    def __init__(self, message):
        mainWindow = tkinter.Tk()
        v = tkinter.StringVar()
        v.set(message)
        label = tkinter.Label(mainWindow, text='Current Activity: ', font=("Helvetica", 60))
        label.pack()
        label2 = tkinter.Label(mainWindow, textvariable=v, font=("Helvetica", 50), fg='green')
        label2.pack()



        mainWindow.mainloop()
        mainWindow.quit()


    def changeActivity(self, mess):
        global v
        v.set(mess)


if __name__ == '__main__':
    w = Window()