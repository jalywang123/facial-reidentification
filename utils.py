import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def get_face_locations(name):
    print(name)
    #face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img = cv2.imread(name)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect faces
    shape = img.shape
    # print(f'image shape: {shape}')
    faces = face_cascade.detectMultiScale(img, 1.5, 6)
    # print(faces)
    for x, y, w, h in faces:
        # print(x, y, w, h)
        #if y+h<shape[0] and x+w<shape[1]:
        img = img[y:y+h, x:x+w]
    # print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    return img
