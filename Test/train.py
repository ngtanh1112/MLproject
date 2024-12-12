import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()


clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print("ádfasdfasdf")
print(acc)


def visualize_knn(X_train, y_train, X_test, y_pred, title="KNN Visualization"):
    # Tạo đồ thị
    plt.figure(figsize=(8,6))
    
    # Vẽ dữ liệu huấn luyện với màu sắc tương ứng với nhãn
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', label='Train Data')
    
    # Vẽ dữ liệu kiểm thử với màu sắc tương ứng với dự đoán
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', marker='*', edgecolors='k', label='Test Data (Predictions)')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

visualize_knn(X_train, y_train, X_test, predictions, "KNN - 4:1")
