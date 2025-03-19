import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mat_tr = scipy.io.loadmat('train_32x32.mat')
mat_te = scipy.io.loadmat('test_32x32.mat')

import numpy as np
import matplotlib.pyplot as plt

import requests                                      # reading data
from io import StringIO

from sklearn.datasets import fetch_openml            # common data set access
from sklearn.preprocessing import StandardScaler     # scaling transform
from sklearn.model_selection import train_test_split # validation tools
from sklearn.metrics import zero_one_loss as J01
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import sklearn.tree as tree
print("working")
# Fix the random seed for reproducibility
# !! Important !! : do not change this
seed = 1234
np.random.seed(seed)

mat_tr = scipy.io.loadmat('train_32x32.mat')
mat_te = scipy.io.loadmat('test_32x32.mat')

i = 10000
j = 1000

X_tr = mat_tr["X"][:,:,:,:i]
y_tr = mat_tr["y"][:i].flatten()

X_te = mat_te["X"][:,:,:,:j]
y_te = mat_te["y"][:j].flatten()

y_tr[y_tr == 10] = 0
y_te[y_te == 10] = 0

X_tr = np.transpose(X_tr, (3, 0, 1, 2))
X_te = np.transpose(X_te, (3, 0, 1, 2))

X_tr = np.mean(X_tr, axis=3)  
X_te = np.mean(X_te, axis=3)

X_tr = X_tr.reshape(i, -1)
X_te = X_te.reshape(j, -1)

tr_errs = []
te_errs = []
depths = range(1,30,4)
for d in depths:
    learner = RandomForestClassifier(n_estimators = 750,
                bootstrap=True, max_features="log2", 
                max_depth = d, random_state = seed, max_samples=100  )

    learner.fit(X_tr,y_tr)

    tr_errs.append(accuracy_score(y_tr, learner.predict(X_tr)))
    te_errs.append(accuracy_score(y_te, learner.predict(X_te)))

plt.figure(figsize=(8,4))
plt.plot(depths, tr_errs)
plt.plot(depths, te_errs)
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend(['Training accuracy','Testing accuracy'],loc='lower right')
plt.title("Effect of Max Depth on accuracy")
plt.show()

tr_errs = []
te_errs = []

estimators = [100,500,750,1500,2000, 3000]
for e in estimators:
    learner = RandomForestClassifier(n_estimators = e,
                bootstrap=True, max_features="log2", 
                max_depth = 25, random_state = seed, max_samples=500  )

    learner.fit(X_tr,y_tr)

    tr_errs.append(accuracy_score(y_tr, learner.predict(X_tr)))
    te_errs.append(accuracy_score(y_te, learner.predict(X_te)))

plt.figure(figsize=(8,4))
plt.plot(estimators, tr_errs)
plt.plot(estimators, te_errs)
plt.xlabel("Amount of trees in forest")
plt.ylabel("Accuracy")
plt.legend(['Training accuracy','Testing accuracy'],loc='lower right')
plt.title("Effect of tree amount on accuracy")
plt.show()


tr_errs = []
te_errs = []
samps = [10,100,250,500,700]
for m in samps:
    learner = RandomForestClassifier(n_estimators = 750,
                bootstrap=True, max_features="log2", 
                max_depth = 25, random_state = seed, max_samples=m  )

    learner.fit(X_tr,y_tr)

    tr_errs.append(accuracy_score(y_tr, learner.predict(X_tr)))
    te_errs.append(accuracy_score(y_te, learner.predict(X_te)))

plt.figure(figsize=(8,4))
plt.plot(samps, tr_errs)
plt.plot(samps, te_errs)
plt.xlabel("Sample size")
plt.ylabel("Accuracy")
plt.legend(['Training accuracy','Testing accuracy'],loc='lower right')
plt.title("Effect of sample size on accuracy")
plt.show()


learner = RandomForestClassifier(n_estimators = 2000,
                bootstrap=True, max_features="log2", 
                max_depth = 25, random_state = seed, max_samples=1750  )

learner.fit(X_tr,y_tr)

print(f'Accuracy: {accuracy_score(y_te, learner.predict(X_te))}')
