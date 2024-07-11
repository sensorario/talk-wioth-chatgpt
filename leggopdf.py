import fitz 
import os

path = os.path.abspath(os.getcwd())
# print(path)

fullpath = path + '/prova.pdf'
# print(fullpath)

doc = fitz.open(fullpath) 

for page in doc: 
   currenttxt = page.get_text() 
   print(' > ' + currenttxt)