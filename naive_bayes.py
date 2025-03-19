import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.io import loadmat

seed = 1234
np.random.seed(seed)
train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

X = np.moveaxis(train_data['X'], -1, 0)
y = train_data['y'].flatten()
X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])

X_hog = np.array([hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in X_gray])

X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=seed)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

accuracy = nb_model.score(X_test, y_test)
print(f"Naïve Bayes Accuracy on SVHN: {accuracy:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Naïve Bayes on SVHN")
plt.show()