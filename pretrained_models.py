# pretrained models
# it is possible to use a pretrained CNN as part of the network to improve the accuracy of a model
# CNN without dense layers only map the presence of features from the input - so a pretrained CNN can be used as the start of a model
# this will mean a good convolutional base before adding out own dense layer classification at the end
# as the pretrained model will have good idea of what features to look for in an image


# fine tuning when using a pretrained model there needs to be tweaking in the final layers for better accuracy in the final classification of the model's specific problem
# fine tuning = leaving the convoluted layers but fixing the final layers. convolutional layer picking up on the basic componetns/features of the image. When using a pretrained model, adjust the final layers so that the model only looks for relevant features to the specific problem
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

# datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# load the data. splitting the data manually into 80% training, 10% testing, 10% validation
# aiming for above 90% accuracy
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)

print(raw_test)
print(raw_train)
print(raw_validation)
# the following line creates a function object to get the labels
get_label_name = metadata.features['label'].int2str

# display 2 images from the dataset
for image, label in raw_train.take(2):
    print("Image shape:", image.shape)
    print("Label:", get_label_name(label.numpy()))
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label.numpy()))
    plt.show()

# this function returns the reshaped image and the label in the reshaped size of 160 by 160
@tf.function(autograph=False)
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    # half of 255 the colour range 
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (160, 160))
    return image, label

# apply the image formatting to all the images using the map function
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# visualising the post-data-processing image data
for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()

plt.close()
    
# shuffle then batch the images
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


# picking a pretrained model
# using the pretrained convolutional base MobileNet V2 developed by Google
# model is trained on 1,400,000 images into 1,000 classes
# only use the base of the model - when loading the model to use, the model will not use the classification layer
# give the pretrained model the input shape and the predeterimined weights from imagenet
IMG_SHAPE = (160, 160, 3)
# include top = is the classifier included with the model
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# output (last layer) of this CNN base is (32, 5, 5, 1280) - feature extraction
# original input layers shape is (1, 160, 160, 3)
# pass this layer into more convolutional layers and a classifier
print(base_model.summary())

# freeze the base = disabling the training property of a layer
# no changes arebeing made to the weights of any layers that are frozen in training
# as the model should not change the convolutional base that already has learned weights and have a high accuracy
# setting the trainable parameters from over 2 million to 0
base_model.trainable = False

# adding a classifier to the pretrained model
# instead of flattening the feature map of the base layer, use a global average pooling layers that will average the entire 5x5 area of each 2d feature map and return a single 1280 element vector per filter
# take the global average of 1280 layers of 5x5 matrices
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# add the prediction layers which will be a single dense neuron, as there are only two prediction classes
prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# 1281 trainable params, 1280 weights between the global average and prediction layer and 1 bias
print(model.summary())

# training the model
# very small learning rate so the model doesnt have any significant  changes made to it
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

initial_epochs = 3
validation_steps = 20

# evaluate the model BEFORE training it using the validation batches
# evaluating just the base pretrained CNN model  
# pre training accuracy is approx 0.55 which is close to a 50/50 guess
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
print("Pre training accuracy: " + str(accuracy0))
print("Pre training loss: " + str(loss0))

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
accuracy = history.history['accuracy']
# accuracy is approx 90% which is very good considering the base laayer classifies over 1000 classes and the model only classifies 2 class (cats and dogs)
print("Accuracy: " + str(accuracy))

# now that the model is create - it has the ability to save a model and then load it
model.save("dogs_vs_cats.h5")
# saving having to retrain the model everytime the script is run
new_model = tf.keras.models.load_model("dogs_vs_cats.h5")

print(new_model)

# object detection and recognition in tensorflow
# github tensorflow API - gives recognition with a score
# there is also a python facial recognition module