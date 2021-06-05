import tensorflow as tf
import os
import random
import numpy
os.environ['CUDA_VISIBLE_DEVICES']='0' # gpu index



def generate_label_from_path(image_path):
    """ Example of randomly generating labeling """
    label = random.randint(0,1)
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

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))



base_path = '~../dog/' # image directory
filenames = os.listdir(base_path)
tfrecord_filename = './test.tfrecord'

with tf.io.TFRecordWriter(tfrecord_filename) as writer:
    for idx, f in enumerate(filenames):
        image_path = os.path.join(base_path, f)
        label = generate_label_from_path(image_path)
        try:
            image_string = open(image_path, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())

        except FileNotFoundError:
            print('File {} could note be found'.format(image_path))
