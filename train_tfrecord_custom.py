import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import datetime
import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# option for parsing tfrecord file
image_feature_description = {
        'height': tf.io.FixedLenFeature([],tf.int64),
        'width': tf.io.FixedLenFeature([],tf.int64),
        'depth': tf.io.FixedLenFeature([],tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }

def read_dataset(batch_size, dataset):
    dataset = dataset.map(_parse_image_function)
    dataset = dataset.map(image_augment)
    dataset = dataset.shuffle(train_data_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(count=eopch_num)
    dataset = dataset.prefetch(2)

    return dataset

def _parse_image_function(example_proto):
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.image.decode_image(features['image_raw'], channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*image_size, 3])
    image.set_shape([*image_size, 3])

    label = tf.cast(features['label'], tf.string)
    label = _parse_label(label,class_num)
    
    return image, label

def _parse_label(label,class_num):
    if label == 'cat':
        label = 0
    else:
        label = 1
    label = tf.one_hot(label, class_num)
    return label

# Data augmentation
def image_augment(image, label):
    image = tf.image.rot90(image,k=1)
    image = tf.image.random_brightness(image, max_delta=60)
    image = tf.image.resize(image, tratget_size, method='bicubic')

    return image, label


# collecting train files 
def config_dataset(train_class,base_dir):
    train_ds_list = []
    valid_ds_list = []
    train_base_dir = os.path.join(base_dir,'train','tfrecord')
    valid_base_dir = os.path.join(base_dir,'valid','tfrecord')

    for class_name in train_class:
        train_ds_list += glob.glob(os.path.join(train_base_dir,class_name,'*.tfrecord'))
        valid_ds_list += glob.glob(os.path.join(valid_base_dir,class_name,'*.tfrecord'))

    print('train count:',len(train_ds_list))
    print('valid count:',len(valid_ds_list))
    train_data_size = len(train_ds_list) + len(valid_ds_list)

    # merging tfrecord dataset
    train_record = tf.data.TFRecordDataset(train_ds_list)
    valid_record = tf.data.TFRecordDataset(valid_ds_list)

    return train_record, valid_record, train_data_size


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

# directory for logging
save_base_dir = os.getcwd()
log_dir = os.path.join(save_base_dir,'log_dir','fit')
save_log_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=save_log_dir)

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
    output = tf.keras.layers.Dense(2,activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[inputs],outputs=[output])

    model.compile(
                  loss = 'binary_crossentropy',
                  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics = ['accuracy']
    )

    return model


if __name__ == '__main__':

    # params setting
    image_size = (800,800)
    tratget_size = (456,456)
    batch_size = 8
    eopch_num = 50

    # root directory
    base_dir = '/my_directory'
    train_class=['cat','dog']
    class_num = len(train_class)

    # dataset to use for training
    train_record, valid_record, train_data_size = config_dataset(train_class,base_dir)

    # batch dataset for training
    train_dataset = read_dataset(batch_size, train_record)
    valid_dataset = read_dataset(batch_size, valid_record)

    model = make_model()

    history = model.fit(
        train_dataset,
        epochs=eopch_num,
        validation_data=valid_dataset,
        verbose=1,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
    )
