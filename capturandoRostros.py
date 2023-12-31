import cv2
import os
import imutils
import configparser


config = configparser.RawConfigParser()
config.read('ConfigFile.properties')
details_dict = dict(config.items('PATH_DATA_IMAGE_STORE'))
NAME_PERSON = 'Enter person name'
FOLDER_CREATED = 'Folder was created'
FACE_PATH = '/rostro_{}.jpg'



dataPath = config.get('PATH_DATA_IMAGE_STORE', 'path')
print(NAME_PERSON)
personName =  input()
dataPath =  config.get('PATH_DATA_IMAGE_STORE', 'path')
personPath = dataPath + '/' + personName

if not os.path.exists(personPath):
    print(FOLDER_CREATED, personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + config.get('MODEL_TRAIN', 'model-haarcascade'))
count = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + FACE_PATH.format(count), rostro)
        count = count + 1
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 100:
        break
cap.release()
cv2.destroyAllWindows()
