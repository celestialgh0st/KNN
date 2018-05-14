from sklearn.datasets import load_iris
from scipy.spatial import distance
from sklearn.neighbors.nearest_centroid import NearestCentroid

def euc(a,b):
	return distance.euclidean(a,b)

class KNN_My():


	def fit(self,x_train,y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self,x_test):
		predictions = []
		for row in x_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self,row):
		best_dist = euc(row,self.x_train[0])
		best_idx = 0
		for i in range(1,len(self.x_train)):
			dist = euc(row,x_train[i])
			if dist < best_dist:
				best_dist = dist
				best_idx = i
		return self.y_train[best_idx]




iris = load_iris()

x =iris.data
y= iris.target

#training data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5)


#training
clf = KNN_My()
clf.fit(x_train,y_train)

#prediction
predictions = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))