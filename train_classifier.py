import datetime
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import settings

test_size_ = settings.DATASET_TEST_RATIO
Tk().withdraw()
dataset_name = askopenfilename()

data_dict = pickle.load(open(dataset_name, 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
print('Разделение датасета в соотношении ' + str(test_size_)+ '.')
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size_, shuffle=True, stratify=labels)

model = RandomForestClassifier()
print('Обучение модели...')
model.fit(x_train, y_train)
print('Модель обучена. Тестирование...')
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% изображений распознано успешно.'.format(score * 100))

model_name = input('Введите название модели, либо нажмите "Enter" для названия по умолчанию: ')
if not model_name.strip():
    model_name = (settings.MODEL_DEFAULT_NAME + str(datetime.datetime.now())).replace(':', '.')
file = open(model_name + '.p', 'wb')
pickle.dump({'model': model}, file)
file.close()
print('Модель ' + model_name + '.p сохранена.')
