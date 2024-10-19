import os
from src.second.functions import pre_process_image

def ler_jpg_para_bmp_e_csv(input_folder, output_folder):
    # Verificar se a pasta de saída existe; caso contrário, criá-la
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Percorrer todos os arquivos na pasta de entrada
    for folder_name in os.listdir(input_folder):
        folder_original = os.path.join(input_folder, folder_name)
        folder_output = os.path.join(output_folder, folder_name)

        # Verificar se a pasta de saída existe; caso contrário, criá-la
        if not os.path.exists(folder_output):
            os.makedirs(folder_output)

        for filename in os.listdir(folder_original):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):  # Verifica se o arquivo é .jpg ou .jpeg
                # Caminho completo do arquivo de entrada
                img_path = os.path.join(folder_original, filename)

                # Converter a imagem para escala de cinza
                img = pre_process_image(img_path)

                output_path = os.path.join(folder_output, os.path.splitext(filename)[0] + '.bmp')
                img.save(output_path)

                print(f'Arquivo {folder_name} - {filename} convertido')

input_folder = './original'
output_folder = './preprocessed'

ler_jpg_para_bmp_e_csv(input_folder, output_folder)

print('Conversões finalizadas')