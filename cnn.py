import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
cnn.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
cnn.add(Flatten())
cnn.add(Dense(units=16, activation='relu'))
cnn.add(Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    r"C:\Users\Dikshant Kumar Singh\OneDrive\Desktop\cnn\train",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    r"C:\Users\Dikshant Kumar Singh\OneDrive\Desktop\cnn\test",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

cnn.fit(
    training_set,
    validation_data=test_set,
    epochs=10
)

image_path = r"C:\Users\Dikshant Kumar Singh\OneDrive\Desktop\cnn\Golden+Retrievers+dans+pet+care.jpeg"
test_image = load_img(image_path, target_size=(64, 64))
image_array = img_to_array(test_image)
image_array = np.expand_dims(image_array, axis=0)
image_array = image_array / 255.0
result = cnn.predict(image_array)
class_labels = training_set.class_indices  # e.g., {'cats': 0, 'dogs': 1}
class_labels_inv = {v: k for k, v in class_labels.items()}  # reverse mapping

if result[0][0] > 0.5:
    predicted_class = class_labels_inv[1]
    confidence = result[0][0]
else:
    predicted_class = class_labels_inv[0]
    confidence = 1 - result[0][0]

plt.imshow(load_img(image_path))
plt.title(f"Predicted: {predicted_class.upper()} (Confidence: {confidence:.2f})")
plt.axis('off')
plt.show()
