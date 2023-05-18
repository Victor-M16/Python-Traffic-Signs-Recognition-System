import numpy as np  # Import the numpy library for numerical operations
import pandas as pd  # Import the pandas library for data manipulation and analysis
import matplotlib.pyplot as plt  # Import the matplotlib library for data visualization
import tensorflow as tf  # Import the TensorFlow library for deep learning operations
from PIL import Image  # Import the Image module from PIL library for image handling
import os  # Import the os module for interacting with the operating system
from sklearn.model_selection import train_test_split  # Import train_test_split from sklearn to split the dataset
from keras.utils import to_categorical  # Import to_categorical from Keras for one-hot encoding
from keras.models import Sequential, load_model  # Import the Sequential and load_model classes from Keras for model creation and loading
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout  # Import the Conv2D, MaxPool2D, Dense, Flatten, Dropout layers from Keras

data = []  # Create an empty list to store the image data
labels = []  # Create an empty list to store the corresponding labels
classes = 43  # Set the number of classes (43 in this case)
cur_path = os.getcwd()  # Get the current working directory path


#Retrieving the images and their labels 
# Loop over the classes
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))  # Create the path to the class directory
    images = os.listdir(path)  # Get the list of images in the class directory

    # Iterate over each image in the class directory
    for a in images:
        try:
            image = Image.open(path + '\\' + a)  # Open the image
            image = image.resize((30, 30))  # Resize the image to (30, 30) pixels
            image = np.array(image)  # Convert the image to a numpy array
            data.append(image)  # Append the image data to the 'data' list
            labels.append(i)  # Append the corresponding label to the 'labels' list
        except:
            print("Error loading image")

data = np.array(data)  # Convert the 'data' list to a numpy array
labels = np.array(labels)  # Convert the 'labels' list to a numpy array

print(data.shape, labels.shape)  # Print the shape of the 'data' and 'labels' arrays

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# Split the data and labels into training and testing sets using train_test_split function

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Print the shapes of the training and testing sets for data and labels

y_train = to_categorical(y_train, 43)  # Convert the training labels to one-hot encoded vectors
y_test = to_categorical(y_test, 43)  # Convert the testing labels to one-hot encoded vectors

model = Sequential()  # Create a sequential model
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
# Add a convolutional layer with 32 filters, kernel size of (5, 5), ReLU activation, and input shape derived from X_train
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))  # Add another convolutional layer
model.add(MaxPool2D(pool_size=(2, 2)))  # Add a max pooling layer with pool size of (2, 2)
model.add(Dropout(rate=0.25))  # Add a dropout layer with a rate of 0.25
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # Add another convolutional layer
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # Add another convolutional layer
model.add(MaxPool2D(pool_size=(2, 2)))  # Add another max pooling layer
model.add(Dropout(rate=0.25))  # Add another dropout layer
model.add(Flatten())  # Flatten the input
model.add(Dense(256, activation='relu'))  # Add a fully connected layer with 256 units and ReLU activation
model.add(Dropout(rate=0.5))  # Add another dropout layer
model.add(Dense(43, activation='softmax'))  # Add a final fully connected layer with 43 units and softmax activation

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Compile the model with categorical cross-entropy loss, Adam optimizer, and accuracy metric

epochs = 15  # Set the number of epochs
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
# Train the model using the training data, batch size of 32, and validation data

model.save("my_model.h5")  # Save the trained model to a file


#plotting graphs for accuracy 
plt.figure(0)  # Create a new figure for plotting (Figure 0)
plt.plot(history.history['accuracy'], label='training accuracy')  # Plot the training accuracy values over epochs
plt.plot(history.history['val_accuracy'], label='val accuracy')  # Plot the validation accuracy values over epochs
plt.title('Accuracy')  # Set the title of the plot to 'Accuracy'
plt.xlabel('epochs')  # Set the label for the x-axis to 'epochs'
plt.ylabel('accuracy')  # Set the label for the y-axis to 'accuracy'
plt.legend()  # Display a legend in the plot
plt.show()  # Display the plot

plt.figure(1)  # Create a new figure for plotting (Figure 1)
plt.plot(history.history['loss'], label='training loss')  # Plot the training loss values over epochs
plt.plot(history.history['val_loss'], label='val loss')  # Plot the validation loss values over epochs
plt.title('Loss')  # Set the title of the plot to 'Loss'
plt.xlabel('epochs')  # Set the label for the x-axis to 'epochs'
plt.ylabel('loss')  # Set the label for the y-axis to 'loss'
plt.legend()  # Display a legend in the plot
plt.show()  # Display the plot

# Testing accuracy on test dataset
from sklearn.metrics import accuracy_score

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))  # Convert the image to a numpy array and append it to the 'data' list

X_test = np.array(data)

pred = model.predict_classes(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))  # Calculate and print the accuracy score with the test data