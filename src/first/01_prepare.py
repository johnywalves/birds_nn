import os
import numpy as np
from PIL import Image
from src.second.functions import pre_process_image

def ler_jpg_para_bmp_e_csv(input_folder, output_folder):
    if not os.path.exists('data'):
        os.makedirs('data')

    # Gerar o arquivo CSV para escrita
    file = open('./data/birds.csv', mode='w', newline='')
    
    # Verificar se a pasta de saída existe; caso contrário, criá-la
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Percorrer todos os arquivos na pasta de entrada
    for folder_name in os.listdir(input_folder):
        folder_original = os.path.join(input_folder, folder_name)
        folder_grayscale = os.path.join(output_folder, folder_name)

        # Verificar se a pasta de saída existe; caso contrário, criá-la
        if not os.path.exists(folder_grayscale):
            os.makedirs(folder_grayscale)

        for filename in os.listdir(folder_original):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):  # Verifica se o arquivo é .jpg ou .jpeg
                # Caminho completo do arquivo de entrada
                img_path = os.path.join(folder_original, filename)

                # Converter a imagem para escala de cinza
                img = pre_process_image(img_path)

                # Definir o caminho de saída
                output_path = os.path.join(folder_grayscale, os.path.splitext(filename)[0] + '.bmp')

                # Salvar a imagem convertida no formato .bmp
                img.save(output_path)

                # Converter a imagem em um array NumPy para manipular os pixels
                img_array = np.array(img).astype(str).flatten()

                # Gerar texto com a linha dos dados em formato CSV
                scale = ','.join(img_array)
                row = f'{folder_name},{scale}\n'

                # Para cada linha de pixels, escrever no arquivo CSV
                file.write(row)

                print(f'Arquivo {folder_name} {filename} convertido')

input_folder = './original'
output_folder = './grayscale'
ler_jpg_para_bmp_e_csv(input_folder, output_folder)
print('Conversões finalizadas')