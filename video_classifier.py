import cv2
import mediapipe as mp
import numpy as np
import pickle

# model_name = input('Enter model name: ')
model_name = 'rus_hand2_model'
model = pickle.load(open(model_name + '.p', 'rb'))['model']

video_source = 0
fps = 30
color = (255, 0, 0)

capture = cv2.VideoCapture(video_source)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# модуль умирает, когда находит две руки одновременно, поэтому введен костыль, скелет с метками строится максимум для одной руки
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=1)

label_dict = {0: 'A', 1: 'B'}

while True:
    ret, frame = capture.read()

    H, W, ch = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # mediapipe кушает только RGB, а imread возвращает BGR

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

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, text, (x2 + 5, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, cv2.LINE_AA)

    cv2.imshow('Video source ' + str(video_source), frame)
    if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
