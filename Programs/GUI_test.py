# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:41:04 2020

@author: User
"""

from tkinter import *



if __name__=="__main__":
    
    root = Tk()
    
    def myclick():
        print("test")
        myLabel = Label(root,text=e_1.get())
        myLabel.pack()
        pass
    
    """
    #label_1 = Label(root, text="here")
    #label_1.pack()
    """
    
    """
    topframe = Frame(root)
    bottomframe = Frame(root)
    topframe.pack(side=TOP)
    bottomframe.pack(side=BOTTOM)
    
    button_1 = Button(bottomframe, text="button 1",fg="red")
    button_2 = Button(topframe, text="button 2",fg="blue")
    button_3 = Button(topframe, text="button 3",fg="green")
    button_4 = Button(topframe, text="button 4",fg="purple")
    
    button_1.pack()
    button_2.pack(side=LEFT)
    button_3.pack(side=TOP)
    button_4.pack()
    """
    """
    one = Label(root,text="one",bg="red",fg="white")
    one.pack()
    
    two = Label(root,text="two",bg="blue",fg="white")
    two.pack(fill=X)
    
    three = Label(root,text="three",bg="green",fg="white")
    three.pack(side=LEFT,fill=Y)
    """
    button_1 = Button(root,text="submit",command=myclick)
    button_1.pack()
    
    e_1 = Entry(root,width = 50,bg = "blue",fg = "white",borderwidth=10)
    e_1.pack()
    
    root.mainloop()
    
    pass