import numpy as np
import glob
import PyPDF2
from PyPDF2 import PdfFileMerger


path = r"C:\data\Rotation_paper\Figures\merge"
		

file_list = glob.glob(path+"\*.h5")

merger = PdfFileMerger()

for pdf in pdfs:
    merger.append(file_list)

merger.write("result.pdf")
