import numpy as np
from pca import PCA_scratch
from knn import KNN_scratch
 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
 
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# PCA
pca = PCA_scratch(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
 
# KNN
knn = KNN_scratch(k=5, metric='euclidean')
knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)
