from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Tiny dummy model
model = Sequential([
    Flatten(input_shape=(224, 224, 3)),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')  # 6 classes
])

# Save model
model.save("temp_model.h5")
print("Temporary model created!")