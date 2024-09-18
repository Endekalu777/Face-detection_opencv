import cv2
import os

cascade_path = 'haarcascade_frontalface_default.xml'

if not os.path.isfile(cascade_path):
    raise FileNotFoundError(f"The file {cascade_path} does not exist.")

face_cascade = cv2.CascadeClassifier(cascade_path)
cap = cv2.VideoCapture(0)

if face_cascade.empty():
    raise Exception(f"Error loading cascade classifier xml file: {cascade_path}")

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("Face detection", img)
    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


