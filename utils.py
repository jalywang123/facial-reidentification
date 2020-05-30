import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_face_locations(name):
    img = cv2.imread(name, 1)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for x, y, w, h in faces:
        img = img[y:y+h, x:x+w]
    
    img = Image.fromarray(img)
    
    return img