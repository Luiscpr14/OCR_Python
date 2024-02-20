import pytesseract
from PIL import Image

img_file = "data/Imagen1.jpg"
im_bw = "temp/bw_image.jpg"

img = Image.open(im_bw)

ocr_result = pytesseract.image_to_string(img)

print(ocr_result)