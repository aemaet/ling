import pandas
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def sti(x):
	if x == "male": return 0
	else: return 1


data = pandas.read_csv('titanic.csv', index_col='PassengerId')

r = data['Age'].notnull()
surv = data[r]["Survived"].as_matrix()
data['Sex'] = data['Sex'].apply(sti)
data = data[r][['Pclass','Fare','Age','Sex']].as_matrix()
#X = np.array([[1, 2], [3, 4], [5, 6]])
#y = np.array([0, 1, 0])
print data
clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, surv)
print clf.feature_importances_