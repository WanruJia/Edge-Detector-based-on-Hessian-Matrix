from tkinter import *
import tkinter, tkinter.filedialog
import csv
import xlwt

root = Tk()
filez = tkinter.filedialog.askopenfilenames(initialdir = "/",title = "Select file",filetypes = (("txt files","*.txt"),("all files","*.*")))
list = root.tk.splitlist(filez)

book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Retention Sum")
sheet1.write(1, 0, "c/20 c")
sheet1.write(2, 0, "c/20 dc")
sheet1.write(3, 0, "1stc/5")
sheet1.write(4, 0, "20th c/5")
sheet1.write(5, 0, "50 c/5")

index =0
for item in list:
    print(item)
    path = item.split('/')
    file = path[-1]
    itemname=file[:-4]
    index = index +1
    sheet1.write(0, index, itemname)
    with open(item, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        rown=0
        for row in spamreader:
            rown=rown+1
            if rown == 4:
                data = row[0].split(',')
                sheet1.write(1, index, float(data[2]))
                sheet1.write(2, index, float(data[4]))
            if rown == 6:
                data = row[0].split(',')
                sheet1.write(3, index, float(data[4]))
            if rown == 25:
                data = row[0].split(',')
                sheet1.write(4, index, float(data[4]))
            if rown == 55:
                data = row[0].split(',')
                sheet1.write(5, index, float(data[4]))
            if rown > 55:
                break
    csvfile.close()

book.save("Retention Sum.xls")
