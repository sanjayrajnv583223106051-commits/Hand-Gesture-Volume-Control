import cv2
import mediapipe as mp
import numpy as np
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

#AUDIO SETUP
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol, _ = volume.GetVolumeRange()

# CAMERA & MEDIAPIPE
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

# GRAPH SETUP 
plt.ion()
fig, ax = plt.subplots(figsize=(4, 4))

volume_history = deque(maxlen=80)
line, = ax.plot([], [], linewidth=2)

ax.set_ylim(0, 100)
ax.set_xlim(0, 80)
ax.set_title("Real-Time Volume Graph")
ax.set_xlabel("Frame")
ax.set_ylabel("Volume (%)")
fig.canvas.draw()

# LOCK VARIABLES 
volume_locked = False
locked_volume = 0

lock_counter = 0
unlock_counter = 0

LOCK_FRAMES = 10      # strong lock
UNLOCK_FRAMES = 5    # fast unlock

#FPS
pTime = 0

# FINGER COUNT FUNCTION
def count_fingers(lmList):
    fingers = []

    fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(1 if lmList[tip][2] < lmList[tip - 2][2] else 0)

    return sum(fingers)

# MAIN LOOP
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, info in enumerate(results.multi_handedness):
            if info.classification[0].label == "Left":
                left_hand = results.multi_hand_landmarks[i]
            else:
                right_hand = results.multi_hand_landmarks[i]

    #LEFT HAND (VOLUME CONTROL)
    if left_hand and not volume_locked:
        mpDraw.draw_landmarks(img, left_hand, mpHands.HAND_CONNECTIONS)

        lmList = []
        h, w, _ = img.shape
        for id, lm in enumerate(left_hand.landmark):
            lmList.append([id, int(lm.x * w), int(lm.y * h)])

        if len(lmList) > 8:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 4)

            distance = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(distance, [30, 200], [minVol, maxVol])
            volPercent = np.interp(distance, [30, 200], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)
            volume_history.append(volPercent)

            cv2.putText(img, f"{int(volPercent)}%",
                        (30, 70), cv2.FONT_HERSHEY_COMPLEX,
                        1.2, (0, 0, 0), 3)

    #RIGHT HAND (LOCK / UNLOCK)
    if right_hand:
        mpDraw.draw_landmarks(img, right_hand, mpHands.HAND_CONNECTIONS)

        lmList = []
        h, w, _ = img.shape
        for id, lm in enumerate(right_hand.landmark):
            lmList.append([id, int(lm.x * w), int(lm.y * h)])

        finger_count = count_fingers(lmList)

        #LOCK: fist or near fist (0 or 1 finger) 
        if finger_count <= 1:
            lock_counter += 1
            unlock_counter = 0

            if lock_counter >= LOCK_FRAMES and not volume_locked:
                volume_locked = True
                locked_volume = volume.GetMasterVolumeLevel()

        # UNLOCK: open palm (4 or 5 fingers)
        elif finger_count >= 4:
            unlock_counter += 1
            lock_counter = 0

            if unlock_counter >= UNLOCK_FRAMES and volume_locked:
                volume_locked = False

        else:
            lock_counter = 0
            unlock_counter = 0

    # STATUS DISPLAY
    status = "LOCKED 🔒" if volume_locked else "UNLOCKED 🔓"
    color = (0, 0, 255) if volume_locked else (0, 200, 0)
    cv2.putText(img, status, (430, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, color, 3)

    # GRAPH UPDATE 
    line.set_xdata(range(len(volume_history)))
    line.set_ydata(list(volume_history))
    ax.set_xlim(0, len(volume_history))
    fig.canvas.draw()
    fig.canvas.flush_events()

    buf = fig.canvas.tostring_argb()
    argb = np.frombuffer(buf, dtype=np.uint8)
    argb = argb.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    graph_img = cv2.resize(argb[:, :, 1:], (500, 480))

    #  FPS
    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime
    cv2.putText(img, f"FPS: {fps}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    combined = np.hstack((cv2.resize(img, (640, 480)), graph_img))
    cv2.imshow("Gesture Volume Controller", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
