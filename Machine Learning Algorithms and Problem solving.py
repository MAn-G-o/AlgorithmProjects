############### Machine Learning Algorithms and Problem solving

################# Linear Regression solving Regression problem

import numpy as np
import matplotlib.pyplot as plt

# Define the cost function
def cost_function(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    cost = (1/(2*m)) * np.sum((h-y)**2)
    return cost

# Define the gradient descent function
def gradient_descent(X, y, theta, alpha, n_iterations):
    m = len(y)
    costs = []
    for i in range(n_iterations):
        h = X.dot(theta)
        error = h - y
        grad = (1/m) * X.T.dot(error)
        theta = theta - alpha * grad
        cost = cost_function(X, y, theta)
        costs.append(cost)
    return theta, costs

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Initialize the parameters and hyperparameters
theta = np.random.randn(2, 1)
alpha = 0.1
n_iterations = 1000

# Train the model
theta, costs = gradient_descent(X_b, y, theta, alpha, n_iterations)

# Print the learned parameters
print("Learned parameters:")
print("theta_0 =", theta[0][0])
print("theta_1 =", theta[1][0])

# Plot the data and the linear regression line
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.show()

# Plot the cost function vs. iterations
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations")
plt.show()


########################### logistic regression solving binary classification problem 

import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    return cost

# Define the gradient descent function
def gradient_descent(X, y, theta, alpha, n_iterations):
    m = len(y)
    costs = []
    for i in range(n_iterations):
        h = sigmoid(X.dot(theta))
        error = h - y
        grad = (1/m) * X.T.dot(error)
        theta = theta - alpha * grad
        cost = cost_function(X, y, theta)
        costs.append(cost)
    return theta, costs

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 2) - 1
y = (X[:,0] + X[:,1] > 0).astype(int)

# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Initialize the parameters and hyperparameters
theta = np.zeros((3, 1))
alpha = 0.1
n_iterations = 1000

# Train the model
theta, costs = gradient_descent(X_b, y, theta, alpha, n_iterations)

# Print the learned parameters
print("Learned parameters:")
print("theta_0 =", theta[0][0])
print("theta_1 =", theta[1][0])
print("theta_2 =", theta[2][0])

# Plot the data and the decision boundary
plt.scatter(X[:,0], X[:,1], c=y)
x1 = np.linspace(-1, 1, 100)
x2 = -(theta[0][0] + theta[1][0]*x1) / theta[2][0]
plt.plot(x1, x2, color='red')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Logistic Regression")
plt.show()

# Plot the cost function vs. iterations
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations")
plt.show()

################################### classification problem and a regression problem using decision trees

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification problem: predicting breast cancer diagnosis (malignant or benign)
# Load the breast cancer dataset
cancer = load_breast_cancer()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing set
score = clf.score(X_test, y_test)
print(f"Accuracy score: {score:.3f}")

# Regression problem: predicting Boston housing prices
# Load the Boston housing dataset
boston = load_boston()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)

# Train a decision tree regressor
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)

# Evaluate the regressor on the testing set
score = reg.score(X_test, y_test)
print(f"R-squared score: {score:.3f}")


###################################### solving a classification problem and a regression problem using random forests

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification problem: predicting breast cancer diagnosis (malignant or benign)
# Load the breast cancer dataset
cancer = load_breast_cancer()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing set
score = clf.score(X_test, y_test)
print(f"Accuracy score: {score:.3f}")

# Regression problem: predicting Boston housing prices
# Load the Boston housing dataset
boston = load_boston()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)

# Train a random forest regressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# Evaluate the regressor on the testing set
score = reg.score(X_test, y_test)
print(f"R-squared score: {score:.3f}")


######################################### solving a classification problem and a regression problem using SVM

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR

# Classification problem: predicting breast cancer diagnosis (malignant or benign)
# Load the breast cancer dataset
cancer = load_breast_cancer()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear', C=1, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing set
score = clf.score(X_test, y_test)
print(f"Accuracy score: {score:.3f}")

# Regression problem: predicting Boston housing prices
# Load the Boston housing dataset
boston = load_boston()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)

# Train an SVM regressor
reg = SVR(kernel='linear', C=1, epsilon=0.1)
reg.fit(X_train, y_train)

# Evaluate the regressor on the testing set
score = reg.score(X_test, y_test)
print(f"R-squared score: {score:.3f}")
