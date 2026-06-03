# webcam_test.py

import cv2

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()