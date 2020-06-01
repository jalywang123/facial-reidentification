import shutil
import os
DATASET_PATH='./lfw_funneled'
print(f'Length of dataset before sanitizing: {len(os.listdir(DATASET_PATH))}')
count = 0
for i in os.listdir(DATASET_PATH):
    if len(os.listdir(f'{DATASET_PATH}/{i}')) == 1:
        shutil.rmtree(f'{DATASET_PATH}/{i}')

print(f'Length of dataset after sanitizing: {len(os.listdir(DATASET_PATH))}')
