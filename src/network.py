import tensorflow as tf
from tensorflow import keras

SMOOTH = 1.

# network blocks
def bn_activation(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act:
        x = keras.layers.Activation("relu")(x)
    return x

def convolution_block(x, filters, kernel_size=(3,3), padding="same", strides=1):
    conv = bn_activation(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3,3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = convolution_block(x, filters, kernel_size, padding, strides)

    shortcut = keras.layers.Conv2D(filters, kernel_size=(1,1), padding=padding, strides=strides)(x)
    shortcut = bn_activation(shortcut, False)

    output = keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = convolution_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = convolution_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_activation(shortcut, act=False)
    
    output = keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    c = keras.layers.Concatenate()([u, xskip])
    return c


def ResUNet(image_size, num_classes):
    #f = [16, 32, 64, 128, 256]
    f = [24, 48, 96, 192, 384]
    inputs = keras.layers.Input((image_size, image_size, 1))
    
    ## Encoder
    e0 = inputs # (192, 192, 1)
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = convolution_block(e5, f[4], strides=1)
    b1 = convolution_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = keras.layers.Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(d4)

    model = keras.models.Model(inputs, outputs)
    return model

def dice_coef(y_true, y_pred):
    print(y_pred.shape)
    y_true = keras.utils.to_categorical(y_true, num_classes=4)
    y_true = tf.compat.v1.constant(y_true, shape=[192, 192, 4])
    y_true_f = tf.compat.v1.layers.flatten(y_true)
    print("gt: ", y_true_f.shape)
    y_pred_f = tf.compat.v1.layers.flatten(y_pred)
    print("pred: ", y_pred_f.shape)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def build_ResUNet(image_size, num_classes):
    model = ResUNet(image_size, num_classes)
    adam = keras.optimizers.Adam()
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=adam, loss=loss_function, metrics=['accuracy'])
    model.summary()
    return model
