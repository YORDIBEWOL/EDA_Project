import tensorflow as tf
import os
import numpy as np
np.random.seed(1)
tf.random.set_seed(2)
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#from tensorflow.keras.applications.efficientnet import EfficientNetB0
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import CSVLogger


csv_logger = CSVLogger('log.csv', append=True, separator=';')

batch_size = 32 # 32 examples in a mini-batch, smaller batch size means more updates in one epoch
num_classes = 10#
epochs = 3# repeat 200 times
data_augmentation = True

input_shape = (32,32,3)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
base_model  = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
#base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
#base_model = tf.keras.applications.ResNet101(include_top=False, weights="imagenet", input_shape=input_shape)
#base_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
#base_model = tf.keras.applications.MobileNet(include_top=False, weights="imagenet", input_shape=input_shape)

#import pdb; pdb.set_trace()
initializer = tf.keras.initializers.GlorotUniform(seed=1)
regularizer = tf.keras.regularizers.l2(0.0001) # 0.0001
global_average_layer = layers.GlobalAveragePooling2D()
flatten_layer = layers.Flatten()
dense_layer = layers.Dense(10, use_bias=False, kernel_initializer=initializer, bias_initializer='zeros', name='Bottleneck', activation='softmax')

Model  = tf.keras.Sequential([base_model, global_average_layer, flatten_layer, dense_layer])
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255

sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)
Model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
cnn = Model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test), shuffle=False,callbacks=[csv_logger])
