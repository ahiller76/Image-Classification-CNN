# -*- coding: utf-8 -*-

### Aaron Hiller, Kit Sloan

import keras
from keras import layers
from pathlib import Path
from keras import models
import os, shutil
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


# Finds directory used for python file
Main_Dir = os.getcwd()

# Path to data
Data = os.path.join(Main_Dir, "flower_photos")
Data_Daisy = os.path.join(Data, "daisy")
Data_Dandelion = os.path.join(Data, "dandelion")
Data_Rose = os.path.join(Data, "roses")
Data_Sunflower = os.path.join(Data, "sunflowers")
Data_Tulip = os.path.join(Data, "tulips")

# Path to training
Training = os.path.join(Main_Dir, "Training")
Train_Daisy = os.path.join(Training, "daisy")
Train_Dandelion = os.path.join(Training, "dandelion")
Train_Rose = os.path.join(Training, "roses")
Train_Sunflower = os.path.join(Training, "sunflowers")
Train_Tulip = os.path.join(Training, "tulips")

# Path to validation
Validating = os.path.join(Main_Dir, "Validating")
Val_Daisy = os.path.join(Validating, "daisy")
Val_Dandelion = os.path.join(Validating, "dandelion")
Val_Rose = os.path.join(Validating, "roses")
Val_Sunflower = os.path.join(Validating, "sunflowers")
Val_Tulip = os.path.join(Validating, "tulips")

# Path to test
Testing = os.path.join(Main_Dir, "Testing")
Test_Daisy = os.path.join(Testing, "daisy")
Test_Dandelion = os.path.join(Testing, "dandelion")
Test_Rose = os.path.join(Testing, "roses")
Test_Sunflower = os.path.join(Testing, "sunflowers")
Test_Tulip = os.path.join(Testing, "tulips")

# rename data daisy files
os.chdir(Data_Daisy)
D_Daisy_Cnt = 0
for file in os.listdir():
    if file.startswith('daisy') or file.endswith(".h5"):
        break
    else:
        src = file
        dst = 'daisy' + str(D_Daisy_Cnt) + '.jpg'
        os.rename(src, dst)
        D_Daisy_Cnt += 1

# rename data dandelion files
os.chdir(Data_Dandelion)
D_Dandelion_Cnt = 0
for file in os.listdir():
    if file.startswith('dandelion') or file.endswith(".h5"):
        break
    else:
        src = file
        dst = 'dandelion' + str(D_Dandelion_Cnt) + '.jpg'
        os.rename(src, dst)
        D_Dandelion_Cnt += 1

# rename data roses files
os.chdir(Data_Rose)
D_Rose_Cnt = 0
for file in os.listdir():
    if file.startswith('roses'):
        break
    else:
        src = file
        dst = 'roses' + str(D_Rose_Cnt) + '.jpg'
        os.rename(src, dst)
        D_Rose_Cnt += 1

# rename data sunflowers files
os.chdir(Data_Sunflower)
D_Sun_Cnt = 0
for file in os.listdir():
    if file.startswith('sunflowers'):
        break
    else:
        src = file
        dst = 'sunflowers' + str(D_Sun_Cnt) + '.jpg'
        os.rename(src, dst)
        D_Sun_Cnt += 1

# rename data tulips files
os.chdir(Data_Tulip)
D_Tulip_Cnt = 0
for file in os.listdir():
    if file.startswith('tulips') or file.endswith(".h5"):
        break
    else:
        src = file
        dst = 'tulips' + str(D_Tulip_Cnt) + '.jpg'
        os.rename(src, dst)
        D_Tulip_Cnt += 1

# Create directories for the following paths

Path(Train_Daisy).mkdir(parents=True, exist_ok=True)
Path(Train_Dandelion).mkdir(parents=True, exist_ok=True)
Path(Train_Rose).mkdir(parents=True, exist_ok=True)
Path(Train_Sunflower).mkdir(parents=True, exist_ok=True)
Path(Train_Tulip).mkdir(parents=True, exist_ok=True)
Path(Val_Daisy).mkdir(parents=True, exist_ok=True)
Path(Val_Dandelion).mkdir(parents=True, exist_ok=True)
Path(Val_Rose).mkdir(parents=True, exist_ok=True)
Path(Val_Sunflower).mkdir(parents=True, exist_ok=True)
Path(Val_Tulip).mkdir(parents=True, exist_ok=True)
Path(Test_Daisy).mkdir(parents=True, exist_ok=True)
Path(Test_Dandelion).mkdir(parents=True, exist_ok=True)
Path(Test_Rose).mkdir(parents=True, exist_ok=True)
Path(Test_Sunflower).mkdir(parents=True, exist_ok=True)
Path(Test_Tulip).mkdir(parents=True, exist_ok=True)

# Copy files to testing, validating and training for daisy

fnames = ['daisy' + str(i) + '.jpg' for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(Data_Daisy, fname)
    dst = os.path.join(Val_Daisy, fname)
    shutil.copyfile(src, dst)

fnames = ['daisy' + str(i) + '.jpg' for i in range(101, 201)]
for fname in fnames:
    src = os.path.join(Data_Daisy, fname)
    dst = os.path.join(Test_Daisy, fname)
    shutil.copyfile(src, dst)

fnames = ['daisy' + str(i) + '.jpg' for i in range(201, len(os.listdir(Data_Daisy)))]
for fname in fnames:
    src = os.path.join(Data_Daisy, fname)
    dst = os.path.join(Train_Daisy, fname)
    shutil.copyfile(src, dst)

# Copy files to testing, validating and training for dandelion

fnames = ['dandelion' + str(i) + '.jpg' for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(Data_Dandelion, fname)
    dst = os.path.join(Val_Dandelion, fname)
    shutil.copyfile(src, dst)

fnames = ['dandelion' + str(i) + '.jpg' for i in range(101, 201)]
for fname in fnames:
    src = os.path.join(Data_Dandelion, fname)
    dst = os.path.join(Test_Dandelion, fname)
    shutil.copyfile(src, dst)

fnames = ['dandelion' + str(i) + '.jpg' for i in range(201, len(os.listdir(Data_Dandelion)))]
for fname in fnames:
    src = os.path.join(Data_Dandelion, fname)
    dst = os.path.join(Train_Dandelion, fname)
    shutil.copyfile(src, dst)

# Copy files to testing, validating and training for roses

fnames = ['roses' + str(i) + '.jpg' for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(Data_Rose, fname)
    dst = os.path.join(Val_Rose, fname)
    shutil.copyfile(src, dst)

fnames = ['roses' + str(i) + '.jpg' for i in range(101, 201)]
for fname in fnames:
    src = os.path.join(Data_Rose, fname)
    dst = os.path.join(Test_Rose, fname)
    shutil.copyfile(src, dst)

fnames = ['roses' + str(i) + '.jpg' for i in range(201, len(os.listdir(Data_Rose)))]
for fname in fnames:
    src = os.path.join(Data_Rose, fname)
    dst = os.path.join(Train_Rose, fname)
    shutil.copyfile(src, dst)

# Copy files to testing, validating and training for sunflowers

fnames = ['sunflowers' + str(i) + '.jpg' for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(Data_Sunflower, fname)
    dst = os.path.join(Val_Sunflower, fname)
    shutil.copyfile(src, dst)

fnames = ['sunflowers' + str(i) + '.jpg' for i in range(101, 201)]
for fname in fnames:
    src = os.path.join(Data_Sunflower, fname)
    dst = os.path.join(Test_Sunflower, fname)
    shutil.copyfile(src, dst)

fnames = ['sunflowers' + str(i) + '.jpg' for i in range(201, len(os.listdir(Data_Sunflower)))]
for fname in fnames:
    src = os.path.join(Data_Sunflower, fname)
    dst = os.path.join(Train_Sunflower, fname)
    shutil.copyfile(src, dst)

# Copy files to testing, validating and training for tulips

fnames = ['tulips' + str(i) + '.jpg' for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(Data_Tulip, fname)
    dst = os.path.join(Val_Tulip, fname)
    shutil.copyfile(src, dst)

fnames = ['tulips' + str(i) + '.jpg' for i in range(101, 201)]
for fname in fnames:
    src = os.path.join(Data_Tulip, fname)
    dst = os.path.join(Test_Tulip, fname)
    shutil.copyfile(src, dst)

fnames = ['tulips' + str(i) + '.jpg' for i in range(201, len(os.listdir(Data_Tulip)))]
for fname in fnames:
    src = os.path.join(Data_Tulip, fname)
    dst = os.path.join(Train_Tulip, fname)
    shutil.copyfile(src, dst)

# Resize all the photos

size = (100,100)
# Create our CNN model
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# Data preprocessing
# Create the train set with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, 
    vertical_flip=True,)

train_generator = train_datagen.flow_from_directory(
    Training,
    target_size=(150, 150),
    batch_size=128,
    class_mode='categorical')

# Create the validation data set.
valid_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = valid_datagen.flow_from_directory(
        Validating,
        target_size=(150, 150),
        batch_size=128,
        class_mode='categorical')

# Create the test data set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    Testing,
    target_size=(150, 150),
    batch_size=128,
    class_mode='categorical')

# train the model
history = model.fit(
      train_generator,
      steps_per_epoch=50,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=30)

# Save the model.
model.save('flower_classification.h5')

# Plot results:
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Evaludate the model using the test set.
score = model.evaluate(test_generator, steps=30)
print("The test loss is ", score[0])
print("The test accuracy is ", score[1])