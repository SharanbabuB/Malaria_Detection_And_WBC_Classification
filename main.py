from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import tensorflow as tf

# Define batch size and number of epochs
batch_size = 32
n_epochs = 30

# Define data generators for training
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
        'data',  # This is the source directory for training images
        target_size=(400, 400),  # All images will be resized to 400 x 400
        batch_size=batch_size,
        classes=['EOSINOPHIL','LYMPHOCYTE','Malaria Parasitized','Malaria Uninfected','MONOCYTE','NEUTROPHIL'], 
        class_mode='categorical')

# Define the CNN model architecture
model = Sequential([
    Convolution2D(16, (3, 3), activation='relu', input_shape=(400, 400, 3)),
    MaxPooling2D(2, 2),
    Convolution2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Convolution2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Convolution2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Convolution2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')  # Update to 2 classes (Abnormal and Normal)
])

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['acc'])

# Train the model and store the history
history = []
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    epoch_history = model.fit(
                        train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=1,
                        verbose=1)
    history.append(epoch_history)

# Plot training accuracy and loss
plt.figure(figsize=(10, 5))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot([acc for epoch_history in history for acc in epoch_history.history['acc']], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training loss
plt.subplot(1, 2, 2)
plt.plot([loss for epoch_history in history for loss in epoch_history.history['loss']], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Show plots
plt.tight_layout()
plt.show()

# Save the trained model
model.save('model.h5')