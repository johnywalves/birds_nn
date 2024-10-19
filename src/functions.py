from PIL import Image

def pre_process_image(img_path):
    # Abrir a imagem usando PIL
    img = Image.open(img_path)

    # Converter a imagem para escala de cinza
    img_gray = img.convert('L')

    return img_gray