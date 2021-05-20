# USAGE
# python cnn_regression.py --dataset AP

from keras.optimizers import Adam
from keras import losses
from sklearn.model_selection import train_test_split
from pyimagesearch import datasets
from pyimagesearch import models
import numpy as np
import argparse
import locale
import os
import matplotlib.pyplot as plt
from keras import optimizers

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of X-Ray images")
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading size of joint space attributes...")
inputPath = os.path.sep.join([args["dataset"], "XrayGrades.csv"])
df = datasets.load_house_attributes(inputPath)

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading X-Ray images...")
images = datasets.load_house_images(df, args["dataset"])
images = images / 255.0

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(df, images, test_size=0.2, random_state=42,stratify=df['Gender&Side'])
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

testY = testAttrX["Grade"]

# create our Convolutional Neural Network and then compile the model
# using mean absolute percentage error as our loss, implying that we
# seek to minimize the absolute percentage difference between our
# price *predictions* and the *actual prices*
model = models.create_cnn(1024, 1024, 1, regress=False)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

# train the model
print("[INFO] training model...")
# for every epoch,
splitval = train_test_split(trainAttrX, trainImagesX, test_size=0.1, random_state=42)
(trainAttrX, valAttrX, trainImagesX, valImagesX) = splitval

trainY = trainAttrX["Grade"]
valY = valAttrX["Grade"]
history = model.fit(trainImagesX, trainY, validation_data=(valImagesX, valY),
	epochs=200, batch_size=32)
model.summary()
model.save('XRay.model')
# make predictions on the testing data
print("[INFO] predicting size of joint space...")
preds = model.predict(testImagesX)

# compute the difference between the *predicted* joint space sizes and the
# *actual* sizes, then compute the percentage difference and
# the mean squared error
print(testY,preds)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

print("[INFO] test MSE: " + str(MSE))

# plot loss during training
plt.subplot(2,1,1)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()

plt.subplot(2,1,2)
plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.legend()
plt.show()
