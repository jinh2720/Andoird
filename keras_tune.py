import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import kerastuner as kt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# read tfrecord file
train_raw_image_ds = tf.data.TFRecordDataset('train/train.tfrecord')
valid_raw_image_ds = tf.data.TFRecordDataset('valid/valid.tfrecord')

# image_size = (800,800)
image_size = (1024,1024)
tratget_size = (224,224)
batch_size = 4

# option for parsing tfrecord file
image_feature_description = {
        'height': tf.io.FixedLenFeature([],tf.int64),
        'width': tf.io.FixedLenFeature([],tf.int64),
        'depth': tf.io.FixedLenFeature([],tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

def read_dataset(batch_size, dataset):
    dataset = dataset.map(_parse_image_function)
    dataset = dataset.prefetch(10)
    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

def _parse_image_function(example_proto):
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.image.decode_image(features['image_raw'], channels=3)
    # image = tf.io.decode_raw(features['image_raw'], tf.int8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*image_size, 3])
    image.set_shape([*image_size, 3])
    image = tf.image.resize(image, tratget_size, method='bicubic')

    label = tf.cast(features['label'], tf.float32)

    return image, label

# batch dataset for training
train_dataset = read_dataset(batch_size, train_raw_image_ds)
valid_dataset = read_dataset(batch_size, valid_raw_image_ds)

print('train set:',train_dataset)
print('valid set:',train_dataset)


# setting options for training
initial_learning_rate = 1e-5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "./melanoma_model.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    save_weights_only=False,
    mode='auto',
    period=1
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
    min_delta=0,
    verbose=1,
    mode='auto'
)


# tuning target - dropout, learning rate
def make_model(hp):

    base_model = tf.keras.applications.DenseNet201(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None
    )
    base_model.trainable = False
    inputs = tf.keras.layers.Input([*tratget_size, 3])
    x = preprocess_input(inputs)
    x = base_model(x)


    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(x)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[inputs],outputs=[output])

    model.compile(
                  loss = 'binary_crossentropy',
                  optimizer = tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5, 1e-6])),
                  metrics =['accuracy']
    )

    return model


tuner = kt.Hyperband(
    make_model,
    objective='val_accuracy',
    max_epochs=5,
    hyperband_iterations=2)

tuner.search(train_dataset,
             validation_data=valid_dataset,
             epochs=5,
             callbacks=[checkpoint_cb, early_stopping_cb])
