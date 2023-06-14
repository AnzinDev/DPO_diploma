import os
import cv2
import settings

data_dir = settings.DATA_DIRECTORY
number_of_classes = settings.NUMBER_OF_CLASSES
dataset_size = settings.DATASET_SIZE
interface_color = settings.INTERFACE_COLOR_COLLECT_IMGS
font_scale = settings.FONT_SCLAE_COLLECT_IMGS
font_thickness = settings.FONT_THICKNESS_COLLECT_IMGS

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

cap = cv2.VideoCapture(settings.VIDEO_SOURCE_COLLECT_IMGS)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(data_dir, str(j))):
        os.makedirs(os.path.join(data_dir, str(j)))

    print('Сбор изображений для класса {}'.format(j))

    while True:
        ret, frame = cap.read()
        H, W, ch = frame.shape
        cv2.putText(frame, 'Press Q to collect images', (5, H - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    interface_color, font_thickness, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame_show = frame.copy()
        cv2.putText(frame_show, 'Make a gesture', (5, H - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, interface_color,
                    font_thickness, cv2.LINE_AA)
        cv2.imshow('frame', frame_show)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
