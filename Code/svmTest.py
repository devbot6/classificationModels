import numpy as np
import matplotlib.pyplot as plt

# Imports - from scikit-learn
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Setting for generating some random data
n_points = 50             # Total number of data points
label_prop = 0.5          # Proportion of points in class 1

# Initialize data matrix (as zeros)
data = np.zeros(shape=[n_points, 2])

# Set up the number of data points in each class
n_data_1 = int(n_points * label_prop)
n_data_2 = n_points - n_data_1

# Generate the data
data[0:n_data_1, 0] = np.abs(np.random.randn(n_data_1))
data[0:n_data_1, 1] = np.abs(np.random.randn(n_data_1))
data[n_data_2:, 0] = np.abs(np.random.randn(n_data_1)) + 2
data[n_data_2:, 1] = np.abs(np.random.randn(n_data_1)) + 2

labels = np.array([0] * n_data_1 + [1] * n_data_2)

fig = plt.figure(figsize=[9, 7])
plt.plot(data[0:n_data_1, 0], data[0:n_data_1, 1],
         'b.', ms=12, label="Label=0")
plt.plot(data[n_data_2:, 0], data[n_data_2:, 1],
         'g.', ms=12, label="Label=1")
plt.legend(fontsize=15)

classifier = SVC(kernel='linear')

classifier.fit(data, labels)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)

train_predictions = classifier.predict(data)

print(classification_report(train_predictions, labels))

# ////////////////////////////////////////////////////////////////////////////////////////////
# Trained Model^^^^^^^

# Define a new data point, that we will predict a label for
new_point = np.array([[3, 3]])

fig.gca().plot(new_point[0][0], new_point[0][1], '.r', ms=12);
fig

# Predict the class of the new data point
prediction = classifier.predict(new_point)
print('Predicted class of new data point is: {}'.format(prediction[0]))

for row in classifier.support_vectors_:
    fig.gca().plot(row[0], row[1], 'ok', ms=14,  mfc='none')
fig

ax = fig.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = classifier.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--']);

# Redraw figure
fig

# Display the plot
plt.show()