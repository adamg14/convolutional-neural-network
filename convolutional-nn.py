# using CIFAR datasret containing 60,000 images 6,000 in each class of everyday objects
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# load and split the 
# will load it as a dataset object
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalise the pixel data values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# label names
class_names = ['airplanes', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# viewing 1 image
IMAGE_INDEX = 1
# plt.imshow(train_images[IMAGE_INDEX], cmap=plt.cm.binary)
# plt.xlabel(class_names[train_labels[IMAGE_INDEX][0]])
# plt.show()

# CNN Architecture
# A common aritecture for CNN is a stack of Conv2D and MaxPooling2D layers followed by a few denesly connected layers
# stack convolutional and maxPooling layers extract features from the iamge.
# These features are flattened and fed to a densly connected layers to derimine the class of an image based on the presence of features
model = models.Sequential()
# arguments = amount of filters, the sample size (size of the filters), activation function (after applying the dot product - apply an AF to it and then put it in the output feature map), input shape what is going into this layers - only need the input shape in the first layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# max pooling 2x2 samples and stride of 2
# shrink it by a factor of 2 each time
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# above convolution base extracts features

# model summary
# we lose pixels in the first layers due to the amount of samples we can take as no padding is applied
# print(model.summary())

# adding dense layers to help with classification based on the combination of features extracted by the convolutional layer
# flatten the 4 x 4 x 64 into a scalar
model.add(layers.Flatten())
# dense layer - connected to each node of the flattened output of the covolution stack
model.add(layers.Dense(64, activation='relu'))
# 10 nodes in the output layer for the 10 different labels
model.add(layers.Dense(10 ))

print(model.summary())


# compiling and training the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(train_images, train_labels, epochs=4, validation_data=(test_images, test_labels))

# evaluating the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
# accuracy approximately 70%
print("Test Accuracy: " + str(test_accuracy))
print("Test Loss: " + str(test_loss))

# techniques to train CNNs on small datasets
# in situations where you dont have millions of images so we cant pick up all the patterns to pick up in all the different classes
# data augmentation is used to avoid overfitting and creating a larger dataset from smaller ones
# this is performing random transformations on the images within the dataset so the model can generalised better
# these transformations include compressions, rotations, stretches and colour changes
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# the following creats a data generator object that transforms images
# turn a single image within a dataset into multiple different images and then pass it into the model so it generalise better
# parameters equal to the image modification
data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# pick an image from the dataset to undergo a transformation
test_image = train_images[14]
# convert the image to a numpy array
img = image.img_to_array(test_image)
# reshape the image
img = img.reshape((1,) + img.shape)

i = 0

# takes the formatted image, save it as test.jpeg, and repeat with random augmentations defined until it breaks
# want to look at different transformation so the model is aware of some differences giving it a better generalisation when it comes to classification and increasing the accuracy
for batch in data_gen.flow(img, save_prefix='test', save_format='jpeg'):
    plt.figure(i)
    # batch[0] is showing the first image in the batch
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:
        break

plt.show()


