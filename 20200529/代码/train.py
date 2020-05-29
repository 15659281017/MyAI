# encoding:utf-8
'''
用于训练垃圾分类模型
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dimensions of our images.
img_width, img_height = 512, 384

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'
nb_train_samples = 1278
nb_validation_samples = 709
epochs = 32 #30
batch_size = 15 #越大精确度越低，效率越高 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#构建线性叠加模型p44 59 133
model = Sequential()
model.add(Conv2D(32, (3, 3),  input_shape=input_shape, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6))  # 6分类
model.add(Activation('softmax'))  # 采用Softmax

#compile用来在魔性编译时配置损失函数、优化器以及正确率函数
model.compile(loss='categorical_crossentropy',  # 多分类  交叉熵损失函数p45
              optimizer='rmsprop',
              metrics=['accuracy'])


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
# this is the augmentation configuration we will use for testing:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')  # 多分类

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')  # 多分类


filepath="weights-improvement-{epoch:02d}-{accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

#fit用来训练一个固定迭代次数的模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.summary()
#evaluate = model.evaluate(input_shape=input_shape)

