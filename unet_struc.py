import tensorflow as tf

#MODIFY AS NECESSARY BASED ON INPUT IMAGE STRUCTURE
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

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