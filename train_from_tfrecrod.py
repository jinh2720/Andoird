import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial



# read tfrecord file
train_raw_image_ds = tf.data.TFRecordDataset('train/train.tfrecord')
valid_raw_image_ds = tf.data.TFRecordDataset('valid/valid.tfrecord')

image_size = (1024,1024)
batch_size = 4

# option for parsing tfrecord file
image_feature_description = {
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
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*image_size, 3])
    label = tf.cast(features['label'], tf.int32)

    return image, label

# batch dataset for training
train_dataset = read_dataset(batch_size, train_raw_image_ds)
valid_dataset = read_dataset(batch_size, valid_raw_image_ds)


# setting options for training
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "melanoma_model.h5", save_best_only=True
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)


def make_model():
    base_model = tf.keras.applications.Xception(
        input_shape=(*image_size, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.layers.Input([*image_size, 3])
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=tf.keras.metrics.AUC(name="auc"),
    )

    return model


with strategy.scope():
    model = make_model()

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
