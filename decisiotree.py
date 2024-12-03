from matplotlib.pylab import f
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data #featurs
y = iris.target # taget label


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

dt_module = DecisionTreeClassifier(random_state = 42)
dt_module.fit(X_train, y_train)

y_pred = dt_module.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred)
print(f" Accurate of Decision Tree classifier: {accuracy_dt}%")
