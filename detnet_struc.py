import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_addons as tfa

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


model = detnet([128, 128, 3])

#OPTIMIZER SET TO USE AMSGRAD
opt = tf.keras.optimizers.Adam(amsgrad = True)
#LOSS FUNCTION SET TO USE SOFT DICE LOSS DEFINED ABOVE
model.compile(optimizer = opt, loss = soft_dice_loss, metrics = ["accuracy"])
model.summary()