import cv2
import mediapipe as mp
import numpy as np
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import settings

Tk().withdraw()
model = pickle.load(open(askopenfilename(title='Выберите файл модели', initialdir='./', filetypes=[("Model File", "*.p"), ("All Files", "*.*")]), 'rb'))['model']

capture = cv2.VideoCapture(settings.VIDEO_SOURCE_VIDEO_CLASSIFIER)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE_VIDEO_CLASSIFIER, max_num_hands=settings.MAX_NUMBER_OF_HANDS)

label_dict = settings.GESTURE_LABELS_DICTIONARY

while True:
    ret, frame = capture.read()

    H, W, ch = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res = hands.process(frame_rgb)

    coord_pair = []
    x_ = []
    y_ = []

    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in res.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                coord_pair.append(x)
                coord_pair.append(y)

                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)

        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        prediction = model.predict([np.asarray(coord_pair)])

        text = label_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), settings.INTERFACE_COLOR_VIDEO_CLASSIFIER, settings.FONT_THICKNESS_VIDEO_CLASSIFIER)
        cv2.putText(frame, text, (x2 + 5, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, settings.FONT_SCLAE_VIDEO_CLASSIFIER, settings.INTERFACE_COLOR_VIDEO_CLASSIFIER, settings.FONT_THICKNESS_VIDEO_CLASSIFIER, cv2.LINE_AA)

    cv2.imshow('Video source ' + str(settings.VIDEO_SOURCE_VIDEO_CLASSIFIER), frame)
    if cv2.waitKey(1000 // settings.FRAMES_PER_SECOND) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
