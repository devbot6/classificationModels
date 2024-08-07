import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Setting for generating some random data
n_points = 75             # Total number of data points
label_prop = [0.33, 0.33, 0.34]  # Proportions of points in each class

# Initialize data matrix (as zeros)
data = np.zeros(shape=[n_points, 2])

# Set up the number of data points in each class
n_data_1 = int(n_points * label_prop[0])
n_data_2 = int(n_points * label_prop[1])
n_data_3 = n_points - n_data_1 - n_data_2

# Generate the data for class 0
data[0:n_data_1, 0] = np.abs(np.random.randn(n_data_1))
data[0:n_data_1, 1] = np.abs(np.random.randn(n_data_1))

# Generate the data for class 1
data[n_data_1:n_data_1+n_data_2, 0] = np.abs(np.random.randn(n_data_2)) + 2
data[n_data_1:n_data_1+n_data_2, 1] = np.abs(np.random.randn(n_data_2)) + 2

# Generate the data for class 2
data[n_data_1+n_data_2:, 0] = np.abs(np.random.randn(n_data_3)) + 4
data[n_data_1+n_data_2:, 1] = np.abs(np.random.randn(n_data_3)) + 4

# Create the labels vector
labels = np.array([0] * n_data_1 + [1] * n_data_2 + [2] * n_data_3)

# Plot out labelled data
fig = plt.figure(figsize=[9, 7])
plt.plot(data[0:n_data_1, 0], data[0:n_data_1, 1], 'b.', ms=12, label="Label=0")
plt.plot(data[n_data_1:n_data_1+n_data_2, 0], data[n_data_1:n_data_1+n_data_2, 1], 'g.', ms=12, label="Label=1")
plt.plot(data[n_data_1+n_data_2:, 0], data[n_data_1+n_data_2:, 1], 'r.', ms=12, label="Label=2")
plt.legend(fontsize=15)

# Initialize a Decision Tree classifier object
classifier = DecisionTreeClassifier()

# Fit our classification model to our training data
classifier.fit(data, labels)

# Calculate predictions of the model on the training data
train_predictions = classifier.predict(data)

# Print out the performance metrics on the training data
print(classification_report(train_predictions, labels))

# Define a new data point, that we will predict a label for
new_point = np.array([[6, 0]])

# Add our new point to figure, in red, and redraw the figure
fig.gca().plot(new_point[0][0], new_point[0][1], '.y', ms=12)
fig

# Predict the class of the new data point
prediction = classifier.predict(new_point)
print('Predicted class of new data point is: {}'.format(prediction[0]))

# Visualizing the decision boundary
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)

# Redraw figure
fig

# Display the plot
plt.show()
