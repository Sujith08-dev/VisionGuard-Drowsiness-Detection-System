import cv2
import pygame
import os
import csv
import time
from datetime import datetime

# ==========================
# INITIAL SETUP
# ==========================

pygame.mixer.init()

os.makedirs("sleep_logs", exist_ok=True)

LOG_FILE = "sleep_logs/sleep_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event", "Screenshot"])

# ==========================
# LOAD CASCADES
# ==========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

# ==========================
# WEBCAM
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access webcam")
    exit()

# ==========================
# VARIABLES
# ==========================

sleep_counter = 0
sleep_events = 0

SLEEP_THRESHOLD = 60

alarm_playing = False
event_recorded = False

# ==========================
# MAIN LOOP
# ==========================

while True:

    success, frame = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    eyes_found = False

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        if len(eyes) >= 2:
            eyes_found = True

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

    # ==========================
    # STATUS LOGIC
    # ==========================

    if eyes_found:

        sleep_counter = 0
        status = "AWAKE"
        color = (0, 255, 0)

        event_recorded = False

        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    else:

        sleep_counter += 1

        if sleep_counter < SLEEP_THRESHOLD:

            status = "DROWSY"
            color = (0, 255, 255)

        else:

            status = "SLEEP DETECTED!"
            color = (0, 0, 255)

            # Alarm
            if not alarm_playing:

                pygame.mixer.music.load("alarm.wav")
                pygame.mixer.music.play(-1)
                alarm_playing = True

            # Save only once
            if not event_recorded:

                sleep_events += 1

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                image_name = (
                    f"sleep_logs/sleep_{timestamp}.jpg"
                )

                cv2.imwrite(
                    image_name,
                    frame
                )

                with open(
                    LOG_FILE,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        datetime.now(),
                        "Sleep Detected",
                        image_name
                    ])

                event_recorded = True

    # ==========================
    # DISPLAY
    # ==========================

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Counter: {sleep_counter}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Sleep Events: {sleep_events}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "VisionGuard - Sleep Detection System",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

# ==========================
# CLEANUP
# ==========================

cap.release()

cv2.destroyAllWindows()

pygame.mixer.music.stop()

pygame.quit()