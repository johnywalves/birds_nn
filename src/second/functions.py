from PIL import Image
from rembg import remove 

def pre_process_image(img_path):
    img = Image.open(img_path)
    output = remove(img)
    return output