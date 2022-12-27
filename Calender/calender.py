# library
from tkinter import *
import calendar

root = Tk()
root.title("Takvim")

year = 2023
myCal = calendar.calendar(year)
cal_year = Label(root, text=myCal, font="Consolas 10 bold")
cal_year.pack()
root.mainloop()
