import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Making separate datasets for training and testing
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory("Training/",
                                          target_size=(150, 150),
                                          batch_size=32,
                                          class_mode='binary')

test_dataset = test.flow_from_directory("Testing/",
                                        target_size=(150, 150),
                                        batch_size=32,
                                        class_mode='binary')

# Create the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
r = model.fit(train_dataset,
              epochs=10,
              validation_data=test_dataset)

print("Final Training Accuracy:", r.history['accuracy'][-1])
print("Final Validation Accuracy:", r.history['val_accuracy'][-1])

# Save the model
model.save("fire_detection_model.h5")

# Plotting loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()

# Function to predict an image using the trained model
def predict_image(filename):
    img1 = image.load_img(filename, target_size=(150, 150))
    plt.imshow(img1)
    Y = image.img_to_array(img1)
    X = np.expand_dims(Y, axis=0)
    val = model.predict(X)
    print(val)
    if val == 1:
        plt.xlabel("No Fire", fontsize=30)
    elif val == 0:
        plt.xlabel("Fire", fontsize=30)

# Example predictions
predict_image("Testing/fire/abc182.jpg")
predict_image('Testing/fire/abc190.jpg')
predict_image('Testing/nofire/abc346.jpg')
predict_image('Testing/nofire/abc361.jpg')
predict_image('Training/fire/abc011.jpg')
predict_image('Testing/fire/abc172.jpg')
predict_image('Testing/nofire/abc341.jpg')

# Show the plots
plt.show()
