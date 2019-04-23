from tkinter import *
import tkinter, tkinter.filedialog
import xlwt

def retrieve_input():
    input = text.get("1.0","end-1c")
    weights = input.split()
    root = Tk()
    filez = tkinter.filedialog.askopenfilenames(initialdir = "/",title = "Select file",filetypes = (("txt files","*.txt"),("all files","*.*")))
    list = root.tk.splitlist(filez)
    index = 0
    for item in list:
        path = item.split('/')
        file = path[-1]
        itemname=file[:-4]
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet(itemname)
        sheet1.write(0, 0, "Garbage Discharge Capacity")
        sheet1.write(0, 1, "Discharge Capacity")
        sheet1.write(0, 2, "Garbo Charge Capacity")
        sheet1.write(0, 3, "Charge Capacity	cycle")
        sheet1.write(0, 5, "Norm. Dis. Cap")

        object = open(item,"r")

        rown = 1
        for lines in object.readlines()[1:]:
            coln = 0
            #print(lines)
            for cell in lines.split():
                if coln <4:
                    sheet1.write(rown,coln,float(cell)/float(weights[index])*1000)
                else:
                    sheet1.write(rown,coln,float(cell))
                coln = coln +1
            rown = rown + 1
            print(lines)
        index = index +1
        book.save("/".join(path[:-1])+"/"+itemname+".xls")
        object.close

root1 = Tk()
root1.title("Weights Input")

app = Frame(root1)
app.pack()

label = Label(app, text ="Please enter the weights for the cells you are going to select,\n then click 'Input' to choose the files")
label.pack()

text = Text(root1)
text.pack()

button = Button(app, text = "Input",command = lambda: retrieve_input())
button.pack(padx = 10,pady = 10,side = LEFT)

button1 = Button(app, text = "Run",command = root1.quit)
button1.pack(padx = 10, pady = 10, side = RIGHT)

root1.mainloop()
