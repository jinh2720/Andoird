import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# read tfrecord file
train_raw_image_ds = tf.data.TFRecordDataset('train/tfrecord/train.tfrecord')
valid_raw_image_ds = tf.data.TFRecordDataset('valid/tfrecord/valid.tfrecord')

image_size = (800,800)
tratget_size = (224,224)
batch_size = 4

# option for parsing tfrecord file
image_feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

def read_dataset(batch_size, dataset):

    dataset = dataset.map(_parse_image_function)
    dataset = dataset.shuffle(buffer_size=100000, seed=None, reshuffle_each_iteration=None) # lager than full data size 
    dataset = dataset.repeat(count=None) # repeatly data suffling per epoch
    dataset = dataset.batch(batch_size, drop_remainder=True) # Learning in batchs
    dataset = dataset.prefetch(2) # prepare next data in advance 

    return dataset

def _parse_image_function(example_proto):
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.image.decode_image(features['image_raw'], channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*image_size, 3])
    image = tf.image.resize(image, tratget_size, method='bicubic')
    label = tf.cast(features['label'], tf.int32)

    return image, label

# batch dataset for training
train_dataset = read_dataset(batch_size, train_raw_image_ds)
valid_dataset = read_dataset(batch_size, valid_raw_image_ds)

print('Train set:',train_dataset)
print('Valid set:',train_dataset)


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


def make_model():

    base_model = tf.keras.applications.DenseNet201(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(*tratget_size, 3),
        pooling=None
    )
    base_model.trainable = False
    inputs = tf.keras.layers.Input([*tratget_size, 3])
    x = preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.425)(x)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[inputs],outputs=[output])

    model.compile(
                  loss = 'binary_crossentropy',
                  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics = ['accuracy']
    )

    return model


model = make_model()

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset,
    verbose=1,
    batch_size = 4,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
