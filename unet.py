import tensorflow as tf
import os
import random
import numpy as np 
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

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

image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()




inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#ENCODER PATH - OBJECT INFORMATION
#ENCODER BLOCK 1
conv_dropout_block_1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
conv_dropout_block_1 = tf.keras.layers.Dropout(0.1)(conv_dropout_block_1)
conv_dropout_block_1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_dropout_block_1)
pool_block_1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_dropout_block_1)
#ENCODER BLOCK 2
conv_dropout_block_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_block_1)
conv_dropout_block_2 = tf.keras.layers.Dropout(0.1)(conv_dropout_block_2)
conv_dropout_block_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_dropout_block_2)
pool_block_2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_dropout_block_2)
#ENCODER BLOCK 3
conv_dropout_block_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_block_2)
conv_dropout_block_3 = tf.keras.layers.Dropout(0.2)(conv_dropout_block_3)
conv_dropout_block_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_dropout_block_3)
pool_block_3 = tf.keras.layers.MaxPooling2D((2, 2))(conv_dropout_block_3)
 #ENCODER BLOCK 4
conv_dropout_block_4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_block_3)
conv_dropout_block_4 = tf.keras.layers.Dropout(0.2)(conv_dropout_block_4)
conv_dropout_block_4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_dropout_block_4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_dropout_block_4)
#ENCODER BLOCK 5
conv_dropout_block_5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
conv_dropout_block_5 = tf.keras.layers.Dropout(0.3)(conv_dropout_block_5)
conv_dropout_block_5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_dropout_block_5)

#DECODER PATH - LOCALIZATION INFORMATION  
#DECODER BLOCK 6
transpose_block_6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_dropout_block_5)
transpose_block_6 = tf.keras.layers.concatenate([transpose_block_6, conv_dropout_block_4])
conv_dropout_block_6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(transpose_block_6)
conv_dropout_block_6 = tf.keras.layers.Dropout(0.2)(conv_dropout_block_6)
conv_dropout_block_6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_dropout_block_6)
#DECODER BLOCK 7 
transpose_block_9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_dropout_block_6)
transpose_block_9 = tf.keras.layers.concatenate([transpose_block_9, conv_dropout_block_3])
conv_dropout_block_7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(transpose_block_9)
conv_dropout_block_7 = tf.keras.layers.Dropout(0.2)(conv_dropout_block_7)
conv_dropout_block_7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_dropout_block_7)
#DECODER BLOCK 8
transpose_block_8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_dropout_block_7)
transpose_block_8 = tf.keras.layers.concatenate([transpose_block_8, conv_dropout_block_2])
conv_dropout_block_8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(transpose_block_8)
conv_dropout_block_8 = tf.keras.layers.Dropout(0.1)(conv_dropout_block_8)
conv_dropout_block_8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_dropout_block_8)
 #DECODER BLOCK 9
transpose_block_9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv_dropout_block_8)
transpose_block_9 = tf.keras.layers.concatenate([transpose_block_9, conv_dropout_block_1], axis=3)
conv_dropout_block_9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(transpose_block_9)
conv_dropout_block_9 = tf.keras.layers.Dropout(0.1)(conv_dropout_block_9)
conv_dropout_block_9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_dropout_block_9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv_dropout_block_9)

#LOSS FUNC -> BINARY CROSSENTROPY -> EVERYTHING IS OR IS NOT A REGION OF INTEREST
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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


ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.draw()

imshow(np.squeeze(X_test[ix]))
