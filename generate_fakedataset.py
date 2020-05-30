import os
import random
from PIL import Image

img = Image.open(random.choice(os.listdir(".")))
img.show()
