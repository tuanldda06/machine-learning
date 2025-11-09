import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = np.array([[2, 3], [1, 1], [3, 4], [2, 2], [3, 5], [1, 2]])
y = np.array([0, 0, 1, 1, 1, 0])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


rf = RandomForestClassifier(n_estimators=100, random_state=42)


rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)
