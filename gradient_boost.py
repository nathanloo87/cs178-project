import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

seed = 1234
np.random.seed(seed)
train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')
X = np.moveaxis(train_data['X'], -1, 0)
y = train_data['y'].flatten() - 1 # change the indices so that it matches with the array. Index starting at 0 vs 1
X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])

X_hog = np.array([hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in X_gray])

X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=seed)

model = xgb.XGBClassifier(objective="multi:softmax", num_class=10, eval_metric="mlogloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy on SVHN: {acc:.4f}")

importance = model.feature_importances_
plt.figure(figsize=(10, 5))
plt.bar(range(len(importance)), importance)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance in XGBoost Model")
plt.show()