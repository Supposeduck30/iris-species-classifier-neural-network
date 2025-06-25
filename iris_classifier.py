import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# One-hot encode labels (new sklearn API)
ohe = OneHotEncoder(sparse_output=False)
y_onehot = ohe.fit_transform(y.values.reshape(-1, 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.25, stratify=y_onehot, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train with output every 25 epochs
class PrintEveryN(tf.keras.callbacks.Callback):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}, val_loss={logs['val_loss']:.4f}, val_acc={logs['val_accuracy']:.4f}")

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    verbose=0,
    callbacks=[PrintEveryN(25)]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal test accuracy: {test_acc:.3f}")

# Plot accuracy & loss
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Helper emoji dictionary
flowers = {
    "setosa": "🌸",
    "versicolor": "🌷",
    "virginica": "🌺"
}

# Randomly pick 5 samples from test set
indices = np.random.choice(len(X_test), size=5, replace=False)
for i, idx in enumerate(indices, 1):
    sample = X_test[idx].reshape(1, -1)
    true_label_idx = y_test[idx].argmax()
    true_label = iris.target_names[true_label_idx]
    
    pred_probs = model.predict(sample)[0]
    pred_label_idx = pred_probs.argmax()
    pred_label = iris.target_names[pred_label_idx]
    confidence = pred_probs[pred_label_idx]

    print(f"\nSample #{i}:")
    print(f"Measurements (scaled): Sepal L={sample[0][0]:.2f}, Sepal W={sample[0][1]:.2f}, Petal L={sample[0][2]:.2f}, Petal W={sample[0][3]:.2f}")
    print(f"Actual species: {flowers[true_label]} {true_label}")
    print(f"Predicted species: {flowers[pred_label]} {pred_label} (confidence: {confidence:.2f})")

    # Plot prediction probabilities bar chart
    plt.figure(figsize=(5,3))
    plt.bar(iris.target_names, pred_probs, color=['#ff9999','#66b3ff','#99ff99'])
    plt.title(f"Prediction confidence for sample #{i}")
    plt.ylim([0,1])
    plt.ylabel("Probability")
    plt.show()
