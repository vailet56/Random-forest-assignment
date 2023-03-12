Random forests and heterogeneous ensembles are two commonly used techniques in machine learning that can improve the accuracy and robustness of predictive models. In this notebook, we will explore these two methods and demonstrate their application using Python and the scikit-learn library.

Random Forests
Random forests are a type of decision tree ensemble that combine the predictions of multiple decision trees to improve the accuracy and robustness of the model. Unlike a single decision tree that can overfit to the training data, a random forest averages the predictions of many decision trees, each trained on a subset of the data and a subset of the features. This helps to reduce overfitting and improve the generalization performance of the model.

To demonstrate the application of random forests, we will use the famous iris dataset. The iris dataset contains measurements of the sepal length, sepal width, petal length, and petal width of 150 iris flowers, divided equally into three species: setosa, versicolor, and virginica.


First, we will load the data and split it into training and testing sets:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

Next, we will create a random forest classifier with 100 decision trees, each with a maximum depth of 3:

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)


We can then use the trained random forest to make predictions on the test set:

y_pred = rf.predict(X_test)


Finally, we can evaluate the performance of the random forest using metrics such as accuracy, precision, recall, and F1-score:

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

