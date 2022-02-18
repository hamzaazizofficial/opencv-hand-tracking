import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, "Fps: " + str(int(fps)), (5, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255),2)

    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
