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
var_smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5] # first one is default
accuracies =[]
best_accuracy = 0
best_value = 0

for var in var_smoothing_values:
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    y_pred = nb_model.predict(X_test)
    

    accuracy = nb_model.score(X_test, y_test)
    accuracies.append(accuracy)
    print(f"var_smoothing={var}: Accuracy={accuracy:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_value = var

# all variance smoothings
print(f"Best Naïve Bayes Accuracy on SVHN: {accuracy:.4f} using {best_value} variance smoothing")
plt.figure(figsize=(8, 6))
plt.plot(var_smoothing_values, accuracies, marker='o', linestyle='-', color='b')
plt.xscale('log')  # Use a logarithmic scale for the x-axis
plt.xlabel('var_smoothing')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. var_smoothing for GaussianNB')
plt.grid(True)
plt.show()

# best model with confusion matrix
bnb = GaussianNB(var_smoothing=best_value)
bnb.fit(X_train, y_train)
best_y_pred = bnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Naïve Bayes on SVHN")
plt.show()