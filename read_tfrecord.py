import tensorflow as tf
import os
import tfx
# from tfx.utils.dsl_utils import external_input # dsl_utils is deprecated
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
import cv2
import numpy as np

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ['CUDA_VISIBLE_DEVICES']='0' # gpu index


raw_image_dataset = tf.data.TFRecordDataset('./test.tfrecord')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(parse_image_function)

for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy() #  image_raw is expected to 'bytes'
    img = cv2.imdecode(np.frombuffer(image_raw, np.uint8), -1) # img is expected to 'numpy array'
    height = image_features['height'].numpy() # height is expected to 'numpy array'
    width = image_features['width'].numpy() # width is expected to 'numpy array'
    depth = image_features['depth'].numpy() # depth is expected to 'numpy array'
    print(height,width,depth)
