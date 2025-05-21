import numpy as np
import cv2 as cv

# Установление соединения с веб-камерой
cap = cv.VideoCapture(0)

while True:
    # Захват кадра
    ret, frame = cap.read()

    # Отображение кадра
    cv.imshow('Frame', frame)

    # Ждем нажатия клавиши 'q' для выхода
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv.destroyAllWindows()

BODY_PARTS = {"Nose": 0, "Neck": 1,"RShoulder": 2,"RElbow": 3,"RWrist": 4,
              "LShoulder": 5,"LElbow": 6,"LWrist": 7,"RHip": 8,"RKnee": 9,
              "RAnkle": 10,"LHip": 11,"LKnee": 12,"LAnkle": 13,"RAye": 14,
              "LAye": 15,"REar": 16,"LEar": 17,"Beckground": 18}
POSE_PARTS = {["Neck","RShoulder"],["Neck","LShoulder"],["RShoulder","RElbow"],["Neck","RShoulder"]
              ,["Neck","RShoulder"],["Neck","RShoulder"],["Neck","RShoulder"],["Neck","RShoulder"]
              ,["Neck","RShoulder"],["Neck","RShoulder"],["Neck","RShoulder"],["Neck","RShoulder"]
              ,["Neck","RShoulder"],["Neck","RShoulder"],["Neck","RShoulder"]}




