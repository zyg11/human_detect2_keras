from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as k
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from human_detect_2 import faces_loaddata
#一些参数
batch_size=8
epochs=10
nb_classes=4
image_width,image_height=64,64
data,label=faces_loaddata.load_data()
#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，
# 直接调用keras提供的这个函数
label=np_utils.to_categorical(label,nb_classes)
# train_data = data[:1680]
# train_labels = label[:1680]
# validation_labels = label[1680:]
# validation_data = data[1680:]
train_data=data
train_label=label
train_data = train_data.astype('float32')#转化为浮点数和归一化很重要
# 将其归一化
train_data /= 255.0


if k.image_data_format()=="channels_first":
    input_shape=(3,image_width,image_height)
else:
    input_shape=(image_width,image_height,3)

#建立model
def Model():
    model=Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes, activation='softmax'))  # 还是用softmax多分类?

    return model
# sgd=SGD(lr=0.005,momentum=0.9,decay=1e-6,nesterov=True)
model=Model()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy']
              )
# train_datagen=ImageDataGenerator(
#     rescale=1./255,
# )
# test_datagen=ImageDataGenerator(
#     rescale=1./255
# )
# train_generator=train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(image_width,image_height),
#     batch_size=batch_size,
#     class_mode='categorical'#多分类
# )
# validation_generator=test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(image_width,image_height),
#     batch_size=batch_size,
#     class_mode='categorical'
# )
# history=model.fit_generator(
#     train_generator,
#     samples_per_epoch=1884,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=100,
#     class_weight='auto'
#  )
history=model.fit(train_data,label,
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  validation_split=0.1
                  )
model.save('E:/keras_data/face6_model/faces_model_6_1.h5')
#画图函数
def plot_training(history):
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc))
    plt.plot(epochs,acc,'b')
    plt.plot(epochs,val_acc,'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs,loss,'b')
    plt.plot(epochs,val_loss,'r')
    plt.title('Training and validation loss')
    plt.show()
#训练的acc_loss图
plot_training(history)