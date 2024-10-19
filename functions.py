from PIL import Image

def pre_process_image(img_path):
    # Abrir a imagem usando PIL
    img = Image.open(img_path)

    # Converter a imagem para escala de cinza
    img_gray = img.convert('L')
    #img_gray.thumbnail((50, 50), Image.Resampling.LANCZOS)

    return img_gray