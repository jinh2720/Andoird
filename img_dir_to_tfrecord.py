import tensorflow as tf
import os
import random
import numpy

os.environ['CUDA_VISIBLE_DEVICES']='0' # gpu index


def generate_label_from_path(label):
    """ Example of randomly generating labeling """
    # label = random.randint(0,1)
    if label == 'dog':
        label=0
    elif label == 'cat':
        label=1
    elif label == 'cow':
        label=2

    return label

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, label):
    image_shape = tf.image.decode_png(image_string).shape
    # label_to_bytes = label.encode(encoding='utf-8') # label을 string으로 그대로 사용하는 경우

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        # 'label': _bytes_feature(label_to_bytes),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


image_dir = '~../tb' # image directory
image_dir_split = image_dir.split('/')
image_dir_parnt = '/'.join(image_dir_split[:-1]) # base directory
image_class_name = image_dir_split[-1] # class
filenames = os.listdir(image_dir)
tfrecord_dir = os.path.join(image_dir_parnt,'tfrecord',image_class_name)
tfrecord_fext = '.tfrecord'

if not os.path.exists(tfrecord_dir):
    os.makedirs(tfrecord_dir)


for idx, f in enumerate(filenames):
    tfrecord_fn = f.split('.')[0] + tfrecord_fext
    tfrecord_file = os.path.join(tfrecord_dir,tfrecord_fn)
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        image_path = os.path.join(image_dir, f)
        label = generate_label_from_path(image_class_name)
        # label = image_class_name
        try:
            image_string = open(image_path, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())
            print(idx,f)

        except FileNotFoundError:
            print('File {} could note be found'.format(image_path))