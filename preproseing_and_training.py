import glob
import pickle
import random


#data reading.....
pkls = glob.glob('./data/*.pkl')
#print(pkls)
random.shuffle(pkls)
data_list = list()
data_lebel = list()
data_lebel_name = list()
for dt in pkls:
	pickle_in = open(dt,"rb")
	dist_dict = pickle.load(pickle_in)
	#print(dist_dict)
	data_list.append(dist_dict['dist_data'])
	data_lebel.append(dist_dict['class'])
	data_lebel_name.append(dist_dict['class_name'])

class_dict = dict(zip(data_lebel, data_lebel_name))
print(class_dict)
with open('class_dict.pkl', 'wb') as f:
	pickle.dump(class_dict, f)


#train test split
lenf = len(data_list)
tran_data = data_list[:round(lenf*0.75)]
tran_data_lebel = data_lebel[:round(lenf*0.75)]

test_data = data_list[:round(lenf*0.25)]
test_data_lebel = data_lebel[:round(lenf*0.25)]

#implimenting KNN algorithm

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(tran_data, tran_data_lebel)

pred = classifier.predict(test_data)

#Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_data_lebel, pred))
print(classification_report(test_data_lebel, pred))

with open('classifier.pkl', 'wb') as f:
	pickle.dump(classifier, f)

from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors':[3,5,8,10,11]}
gs = GridSearchCV(KNeighborsClassifier(),
				grid,verbose = 1,cv=3,n_jobs = -1)
gs_result = gs.fit(tran_data, tran_data_lebel)

print(gs_result)

print (gs_result.best_score_)
print (gs_result.best_params_)
print (gs_result.best_estimator_)