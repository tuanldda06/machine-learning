import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X1 = np.array([[2, 3], [1, 1], [3, 4], [2, 2], [3, 5], [1, 2]])
y1 = np.array([0, 0, 1, 1, 1, 0])
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.33, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test 1 Accuracy:", accuracy)

np.random.seed(42)
X2_pos = np.random.randn(12, 2) * 0.30 + np.array([2.0, 2.0])
X2_neg = np.random.randn(12, 2) * 0.30 + np.array([0.0, 0.0])
X2 = np.vstack([X2_pos, X2_neg])
y2 = np.hstack([np.ones(12), np.zeros(12)])
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.33, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test 2 Accuracy:", accuracy)