from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Create an instance of the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

# Train the MLPClassifier
mlp.fit(X_train, y_train)

# Plot the training loss curve
plt.plot(mlp.loss_curve_)
plt.title("Training Loss Curve for MLP Classifier")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()