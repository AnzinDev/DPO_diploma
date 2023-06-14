import os
import mediapipe as mp
import cv2
import pickle
import datetime
import settings

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE)

data_directory = settings.DATA_DIRECTORY

print('\nСоздание датасета из загруженных изображений. Ожидайте...')

data = [] # данные о координатах меток
labels = [] # данные о классе метки

for obj_class in os.listdir(data_directory):
    for image_path in os.listdir(os.path.join(data_directory, obj_class)):
        coord_pair = []

        img = cv2.imread(os.path.join(data_directory, obj_class, image_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mediapipe кушает только RGB, а imread возвращает BGR

        res = hands.process(img_rgb)
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    coord_pair.append(x)
                    coord_pair.append(y)

            data.append(coord_pair)
            labels.append(obj_class)

dataset_name = input('Датасет создан. Введите имя файла данных, либо нажмите Enter для названия по умолчанию: ')
if not dataset_name.strip():
    dataset_name = (settings.DATASET_DEFAULT_NAME + str(datetime.datetime.now())).replace(':', '.')
file = open(dataset_name + '.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, file)
file.close()
print('Датасет ' + dataset_name + '.pickle сохранен.')
