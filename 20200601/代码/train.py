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
from plot import plot

img_width, img_height = 512, 384

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'
nb_train_samples = 1278
nb_validation_samples = 709
epochs = 50 #50
batch_size = 15 #15

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#构建线性叠加模型
model = Sequential()
model.add(Conv2D(32, (3, 3),  input_shape=input_shape, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
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

model.compile(loss='categorical_crossentropy',  # 多分类  交叉熵损失函数
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,               #数据放缩，把像素点的值除以255，使之在0到1之间
    shear_range=0.2,                #随机错切变换的角度
    zoom_range=0.2,                 #随机缩放的范围
    horizontal_flip=True)           #水平翻转

val_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle='Ture',             #打乱
    class_mode='categorical')   #多分类

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle='Ture',             #打乱
    class_mode='categorical')   #多分类

filepath="weights-improvement-{epoch:02d}-{accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='accuracy',         #需要监视的值
    verbose=1,                  #1为输出epoch模型的保存信息
    save_best_only=False,       #当设置为True时，将只保存在验证集上性能最好的模型
    save_weights_only=False,    #若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    mode='max',                 #决定性能最佳模型的评判准则
    period=1)                   #checkpoint之间的间隔的epoch数
callbacks_list = [checkpoint]

#使用实时数据增强的批数据对模型进行拟合
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

accuracy = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plot(accuracy, val_acc, loss, val_loss) #绘制折线图

model.summary() #模型的结构属性
