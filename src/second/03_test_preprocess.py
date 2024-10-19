import os
import pandas as pd
from src.second.functions import pre_process_image

input_folder = 'test'
output_folder = 'test_preprocessed'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)

    img = pre_process_image(img_path)

    output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.bmp')
    img.save(output_path)