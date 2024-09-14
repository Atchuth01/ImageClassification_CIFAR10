import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

#Load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Normaloze the pixel values to between 0 and 1
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255


#check image data format
if K.image_data_format() == 'channels_first':
    input_shape = (3, 32, 32)
else:
    imput_shape = (32, 32, 3)

#Create the model
model = Sequential([
    #First Conv layer + Max pooling layer
    Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2,2)),

    #Second Conv Layer + MaxPooling
    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2,2)),

    #Third Conv Layer + MaxPooling
    Conv2D(128, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2,2)),

    #Flatten layer
    Flatten(),

    #First fully connected layer
    Dense(128, activation = 'relu'),

    #Layer to prevent overfitting[0.2 - 0.5]
    Dropout(0.4),

    #oUTPUT LAYER
    Dense(10, activation = 'softmax')

])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Convert clas vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#Train the model
trained_model = model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data = (X_test, y_test))

#evaluate the model
score = model.evaluate(X_test, y_test, verbose = 0)
print(f'Loss : {score[0]}; Accuracy : {score[1]}')

#Load the image
img = tf.io.read_file('bird.jpg') #rename with ur test images [jpeg or png]
#img = tf.image.decode_png(img, channels=3) # decode the image[png]
img = tf.image.decode_jpeg(img, channels=3)  # decode the image[jpeg image]
img = tf.image.resize(img, (32, 32)) #resize the image as training images
img = img/255.0 #Normalizing the image

#converr to 4D tensor for keras
img_tensor = tf.expand_dims(img, 0) #adding batch deminsion

#Make predictions
prediction = model.predict(img_tensor)

#Get the predicted class
predicted_class= tf.argmax(prediction[0])

class_names_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                    4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

predicted_class_name =  class_names_dict[predicted_class.numpy()]

#Printing the image class
print('Predicted Class : ', predicted_class_name.upper())
