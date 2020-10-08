import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_addons as tfa
import os
import random
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt


def dice_coef2(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def soft_dice_loss(y_true, y_pred):
    return 1 - dice_coef2(y_true, y_pred)

def encoder_block(kernel, filters, input, pool=True):
    x = input
    if pool:
        x = layers.MaxPool2D()(x)
    x1 = layers.Conv2D(filters=filters, kernel_size=kernel, activation = "relu", kernel_initializer="he_normal", padding="same")(x)
    x1 = tfa.layers.InstanceNormalization()(x1)
    x2 = layers.Conv2D(filters=filters, kernel_size=kernel, activation = "relu", kernel_initializer="he_normal", padding="same")(x1)
    x2 = tfa.layers.InstanceNormalization()(x2)
    x3 = layers.Conv2D(filters=filters, kernel_size=kernel, activation = "relu", kernel_initializer="he_normal", padding="same")(x2)
    x3 = tfa.layers.InstanceNormalization()(x3)
    x = layers.Add()([x1, x3])
    return x


def decoder_block(kernel, filters, input):
    x = input
    x = layers.UpSampling2D()(x)
    x1 = layers.Conv2D(filters=filters, kernel_size=kernel, activation = "relu", kernel_initializer="he_normal", padding="same")(x)
    x1 = tfa.layers.InstanceNormalization()(x1)
    x2 = layers.Conv2D(filters=filters, kernel_size=kernel, activation = "relu", kernel_initializer="he_normal", padding="same")(x1)
    x2 = tfa.layers.InstanceNormalization()(x2)
    x3 = layers.Conv2D(filters=filters, kernel_size=kernel, activation = "relu", kernel_initializer="he_normal", padding="same")(x2)
    x3 = tfa.layers.InstanceNormalization()(x3)
    x = layers.Add()([x1, x3])
    return x


def detnet(input_shape, kernel=3, filters=16):
    inputs = layers.Input(input_shape)
    contract_1 = encoder_block(kernel, filters, inputs, pool=False)
    contract_2 = encoder_block(kernel, filters * 2, contract_1)
    contract_3 = encoder_block(kernel, filters * 4, contract_2)
    expand_1 = decoder_block(kernel, filters * 2, contract_3)
    expand_2 = decoder_block(kernel, filters, expand_1)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(expand_2)

    model = keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model



seed = 30
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_train[n] = mask   

# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

model = detnet([128, 128, 3])
#OPTIMIZER SET TO USE AMSGRAD
opt = tf.keras.optimizers.Adam(amsgrad = True)
#LOSS FUNCTION SET TO USE SOFT DICE LOSS DEFINED ABOVE
model.compile(optimizer = opt, loss = soft_dice_loss, metrics = ["accuracy"])
model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_lab.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=10, callbacks=callbacks)

idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.draw()

imshow(np.squeeze(X_test[2]))
imshow(np.squeeze(preds_test_t[2]))