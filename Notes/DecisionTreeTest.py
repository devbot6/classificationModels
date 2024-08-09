import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Setting for generating some random data
n_points = 75             
label_prop = [0.33, 0.33, 0.34]  

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

labels = np.array([0] * n_data_1 + [1] * n_data_2 + [2] * n_data_3)

fig = plt.figure(figsize=[9, 7])
plt.plot(data[0:n_data_1, 0], data[0:n_data_1, 1], 'b.', ms=12, label="Label=0")
plt.plot(data[n_data_1:n_data_1+n_data_2, 0], data[n_data_1:n_data_1+n_data_2, 1], 'g.', ms=12, label="Label=1")
plt.plot(data[n_data_1+n_data_2:, 0], data[n_data_1+n_data_2:, 1], 'r.', ms=12, label="Label=2")
plt.legend(fontsize=15)

classifier = DecisionTreeClassifier(criterion='log_loss', splitter='best', max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)

classifier.fit(data, labels)

train_predictions = classifier.predict(data)

print(classification_report(train_predictions, labels))

print()

new_point = np.array([[6, 0 ]])

fig.gca().plot(new_point[0][0], new_point[0][1], '.y', ms=12)
fig

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
