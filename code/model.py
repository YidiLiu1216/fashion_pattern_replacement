
from tensorflow import keras

#model blocks
def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    c = keras.layers.Concatenate()([u, xskip])
    return c


def VGG_c2_block(x, filter_size):
    c1 = keras.layers.Conv2D(filters=filter_size, kernel_size=(3, 3), padding="same", activation="relu")(x)
    c2 = keras.layers.Conv2D(filters=filter_size, kernel_size=(3, 3), padding="same", activation="relu")(c1)
    p = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(c2)
    return p

def VGG_c3_block(x, filter_size):
    c1 = keras.layers.Conv2D(filters=filter_size, kernel_size=(3, 3), padding="same", activation="relu")(x)
    c2 = keras.layers.Conv2D(filters=filter_size, kernel_size=(3, 3), padding="same", activation="relu")(c1)
    c3 = keras.layers.Conv2D(filters=filter_size, kernel_size=(3, 3), padding="same", activation="relu")(c2)
    p = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(c3)
    return p

def ASPP(x):
    '''atrous spatial pyramid pooling'''
    dims = keras.backend.int_shape(x)

    y_pool = keras.layers.AveragePooling2D(pool_size=(
        dims[1], dims[2]))(x)
    y_pool = keras.layers.Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', use_bias=False)(y_pool)
    y_pool = keras.layers.BatchNormalization()(y_pool)
    y_pool = keras.layers.Activation('relu')(y_pool)

    #y_pool = keras.layers.UpSampling2D(tensor=y_pool, size=[dims[1], dims[2]])
    y_pool = keras.layers.UpSampling2D((dims[1], dims[2]))(y_pool)
    y_1 = keras.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', use_bias=False)(x)
    y_1 = keras.layers.BatchNormalization()(y_1)
    y_1 = keras.layers.Activation('relu')(y_1)

    y_6 = keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', use_bias=False)(x)
    y_6 = keras.layers.BatchNormalization()(y_6)
    y_6 = keras.layers.Activation('relu')(y_6)

    y_12 = keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', use_bias=False)(x)
    y_12 = keras.layers.BatchNormalization()(y_12)
    y_12 = keras.layers.Activation('relu')(y_12)


    y = keras.layers.concatenate([y_pool, y_1, y_6, y_12])

    y = keras.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal',  use_bias=False)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    return y

#generator model
#based on resnet18-Unet
def ResUNet(image_size):
    #f = [16, 32, 64, 128, 256]
    # f=[64,128,256,512,1024]
    #f=[32,64,128,256,512]
    f=[image_size//8,image_size//4,image_size//2,image_size,image_size*2]
    inputs = keras.layers.Input((image_size, image_size, 4))

    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    #e5 = residual_block(e4, f[4], strides=2)

    ## Bridge
    b0 = conv_block(e4, f[3], strides=1)
    b1 = conv_block(b0, f[3], strides=1)

    ## Decoder
    #u1 = upsample_concat_block(b1, e4)
    #d1 = residual_block(u1, f[4])

    u2 = upsample_concat_block(b1, e3)
    d2 = residual_block(u2, f[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    #d4 = ASPP(d4)
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    #outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)

    model = keras.models.Model(inputs, outputs)
    return model
smooth = 1.

#discriminator model
#based on VGG16
def discriminator(image_size):
    inputs = keras.layers.Input((image_size, image_size, 1))
    e0 = inputs
    b1 = VGG_c2_block(e0, 64)
    b2 = VGG_c2_block(b1, 128)
    b3 = VGG_c2_block(b2, 128)

    b4 = VGG_c3_block(b3, 256)
    b5 = VGG_c3_block(b4, 512)
    b6 = VGG_c3_block(b5, 512)

    f1 = keras.layers.Flatten()(b6)
    d1 = keras.layers.Dense(units=4096, activation="relu")(f1)
    d2 = keras.layers.Dense(units=4096, activation="relu")(d1)
    outputs = keras.layers.Dense(units=1, activation="sigmoid")(d2)

    model1 = keras.models.Model(inputs, outputs)

    return model1

