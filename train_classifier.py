import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

test_size_ = 0.2
shuffle_ = True

dataset_name = input('\nEnter name of dataset: ')

data_dict = pickle.load(open('./' + dataset_name + '.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
print('Splitting dataset in ' + str(test_size_) + ' ratio...')
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size_, shuffle=shuffle_, stratify=labels)

model = RandomForestClassifier()
print('Training model...')
model.fit(x_train, y_train)
print('Trained. Predicting test data...')
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly.'.format(score * 100))

model_name = input('Enter the model name: ')
f = open(model_name + '.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
print('Model ' + model_name + '.p saved.')
