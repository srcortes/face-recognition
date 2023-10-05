import cv2
import os
import numpy as np
import configparser


config = configparser.RawConfigParser()
config.read('ConfigFile.properties')
details_dict = dict(config.items('PATH_DATA_IMAGE_STORE'))
SEPARADOR = '/'
MSN_WAIT = 'Entrenando...'
FINAL_MSN = 'Modelo almacenado...'
dataPath = config.get('PATH_DATA_IMAGE_STORE', 'path')
peopleList = os.listdir(dataPath)
absolutepath = os.path.abspath(__file__)
print("..." + absolutepath)


labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + SEPARADOR + nameDir

	for fileName in os.listdir(personPath):
		labels.append(label)
		facesData.append(cv2.imread(personPath + SEPARADOR + fileName,0))
	label = label + 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print(MSN_WAIT)
face_recognizer.train(facesData, np.array(labels))
face_recognizer.write(config.get('MODEL_TRAIN', 'model'))
print(FINAL_MSN)