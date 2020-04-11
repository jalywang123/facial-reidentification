import os
import cv2
import random
from PIL import  Image
import albumentations as A

img = Image.open(random.choice(os.listdir('.')))
img.show()