import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync) # Number of replicas : 1


AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "/home/jinh/hdd_1/data/vnn_494/img_split/tfrecord/crv_8_2/cv_0"
BATCH_SIZE = 4
IMAGE_SIZE = [1024, 1024]


""" Data Load """
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/data/*.tfrecord') # 그런데 여기서 class별로 리스트를 나누질 않았음...
VALID_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/valid/data/*.tfrecord')

# TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/tfrecords/test*.tfrec")
print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Validation TFRecord Files:", len(VALID_FILENAMES))
# print("Test TFRecord Files:", len(TEST_FILENAMES))


""" Data Decoding """
def decode_image(image):
    # 이미지 정보가 안들어옴..
    # image = tf.image.decode_jpeg(image, channels=3)
    test_shape = tf.image.decode_png(image).shape
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)
    # image = tf.reshape(image, [*IMAGE_SIZE, 3])
    image = tf.reshape(image, [*IMAGE_SIZE])
    return image


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image_raw": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"image_raw": tf.io.FixedLenFeature([], tf.string),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    # image = decode_image(example["image_raw"])

    return example


#### 참고 ####
# feature = {
#     'height': _int64_feature(image_shape[0]),
#     'width': _int64_feature(image_shape[1]),
#     'depth': _int64_feature(image_shape[2]),
#     # 'label': _int64_feature(label),
#     'label': _bytes_feature(label_to_bytes),
#     'image_raw': _bytes_feature(image_string),
# }


"""Loading 방법 정의"""
def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


# 다른 데이터 세트를 얻기 위해 다음 함수를 정의
def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = decode_image(dataset)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


"""입력 이미지 시각"""
train_dataset = get_dataset(TRAINING_FILENAMES)
valid_dataset = get_dataset(VALID_FILENAMES)
# test_dataset = get_dataset(TEST_FILENAMES, labeled=False)

print('데이터타입: ',type(train_dataset))
print(train_dataset)
# print('뭥미?',len(train_dataset))
image_batch, label_batch = next(iter(train_dataset))

# def show_batch(image_batch, label_batch):
#     plt.figure(figsize=(10, 10))
#     for n in range(25):
#         ax = plt.subplot(5, 5, n + 1)
#         plt.imshow(image_batch[n] / 255.0)
#         if label_batch[n]:
#             plt.title("MALIGNANT")
#         else:
#             plt.title("BENIGN")
#         plt.axis("off")
#
#
# show_batch(image_batch.numpy(), label_batch.numpy())
