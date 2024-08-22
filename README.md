CSC312 Machine Learning Assignment

Overview:
This repository contains the solutions and implementations for the CSC312 Machine Learning assignment. The assignment focuses on understanding and implementing linear regression, both from scratch and using the sklearn library.

Project Structure:

Task 1: Implement Linear Regression using Gradient Descent
Objective: Develop a linear regression model by implementing gradient descent to optimize the cost function.
Key Functions:
gradDesc: Runs gradient descent to minimize the cost function and find the optimal parameters.
predH: Computes the hypothesis for a given set of features.
addX0: Adds the intercept term to the feature matrix.

Task 2: Scaling Features for Improved Performance
Objective: Scale the features of the dataset to improve the performance of the gradient descent algorithm.
Key Functions:
obtainScaledXTest: Scales the test set features using the mean and standard deviation of the training set.
makeAPredictionWithScaling: Predicts output values after scaling the test set features.

Task 3: Using sklearn for Linear Regression
Objective: Implement linear regression using the sklearn library to compare results with the manual implementation.

Key Points:
Fit the model to the dataset using model.fit.
Extract the intercept and coefficients representing the learned parameters.
Requirements
Python 3.x
numpy
scikit-learn

You can install the required packages using:
pip install numpy scikit-learn
Running the Code
Gradient Descent Implementation:

To run the gradient descent implementation, ensure that the functions gradDesc, predH, and addX0 are defined.
Execute the provided code snippets to see the model converge to the optimal parameters.
Making Predictions with Scaled Features:

Implement the makeAPredictionWithScaling function and test it with the given test dataset to see how well the model predicts after feature scaling.
Using sklearn for Linear Regression:

The sklearn implementation can be run by importing the necessary libraries and calling model.fit with the feature matrix and output labels.
Example
import sklearn.linear_model as lm
model = lm.LinearRegression()

# Fit the model using the raw features (no scaling)
model.fit(XX, YY)

# Print the model parameters
print("Intercept (θ0):", model.intercept_)
print("Coefficients (θ1, θ2, ...):", model.coef_)
Results
The manually implemented gradient descent approach provided similar results to sklearn's built-in linear regression, demonstrating a good understanding of the underlying mathematical principles.
The predicted values after scaling were found to be very close to the actual values, showcasing the importance of feature scaling in machine learning models.
Conclusion
This assignment reinforced the understanding of linear regression, feature scaling, and the practical application of these concepts using both manual implementation and a high-level library like sklearn.

License
This project is for educational purposes only and is licensed under the MIT License.
