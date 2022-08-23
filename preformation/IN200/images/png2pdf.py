from importlib.resources import path
from fpdf import FPDF
from PIL import Image
import pathlib

# reference
# https://stackoverflow.com/questions/27327513/create-pdf-from-a-list-of-images
# https://stackoverflow.com/questions/3430372/how-do-i-get-the-full-path-of-the-current-files-directory
# https://stackoverflow.com/questions/43767328/python-fpdf-not-sizing-correctly



path_folder = str(pathlib.Path(__file__).parent.resolve())


pdf = FPDF()
imagelist = [
    "q1.png",
    "q2.png",
    "q3.png",
    "q4.png",
    "q5.png",
    "q6.png",
    "q7.png",
    "q8.png",
    "q9.png",
    "q10.png",
    "q11.png",
    "q12.png",
    "q13.png",
    "q14.png",
    "q15.png",
    "q16.png",
    "q17.png",
    "q18.png",
    "q19.png",
    "q20.png",
    ]

# configuration of PDF output file
output_name = "in200_exercice qcm 1_corrige.pdf"
x = 10
y = 10
w = 0
h = 0

for image in imagelist:
    # print(image)
    path_image  = path_folder + "\\" + image
    path_output = path_folder + "\\" + output_name

    cover = Image.open(path_image)
    width, height = cover.size

    # convert pixel in mm with 1px=0.264583 mm
    width, height = float(width * 0.264583), float(height * 0.264583)

    # given we are working with A4 format size 
    pdf_size = {'P': {'w': 210, 'h': 297}, 'L': {'w': 297, 'h': 210}}

    # get page orientation from image size 
    orientation = 'P'
    # orientation = 'L'
    # orientation = 'P' if width < height else 'L'

    #  make sure image size is not greater than the pdf format size
    x_margin = 10 + x
    y_margin = 10 + y

    width = width if width < pdf_size[orientation]['w'] else pdf_size[orientation]['w'] - x_margin
    height = height if height < pdf_size[orientation]['h'] else pdf_size[orientation]['h'] - y_margin
    print(image + "; w:" + str(width) + ", h: " + str(height))


    pdf.add_page(orientation=orientation)
    pdf.image(path_image,x,y,width,height)
pdf.output(path_output, "F")

print("status: done!")