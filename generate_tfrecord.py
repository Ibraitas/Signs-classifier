"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd

from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    row_labels = {'1.1': 1, '1.2': 2, '1.3.1': 3, '1.3.2': 4, '1.4.1': 5, '1.4.2': 6, '1.4.3': 7, '1.4.4': 8,
                  '1.4.5': 9, '1.4.6': 10, '1.5': 11, '1.6': 12, '1.7': 13, '1.8': 14, '1.9': 15, '1.10': 16,
                  '1.11.1': 17, '1.11.2': 18, '1.12.1': 19, '1.12.2': 20, '1.13': 21, '1.14': 22, '1.15': 23,
                  '1.16.1': 24, '1.16.2': 25, '1.16.3': 26, '1.17': 27, '1.18.1': 28, '1.18.2': 29, '1.18.3': 30,
                  '1.18.4': 31, '1.18.5': 32, '1.18.6': 33, '1.19.1': 34, '1.19.2': 35, '1.20': 36, '1.21': 37,
                  '1.22': 38, '1.23': 39, '1.24': 40, '1.25': 41, '1.26': 42, '1.27': 43, '1.28': 44, '1.29': 45,
                  '1.30': 46, '1.31.1': 47, '1.31.2': 48, '1.31.3': 49, '1.31.4': 50, '1.31.5': 51, '1.32': 52,
                  '1.33': 53, '1.34': 54, '1.35': 55, '1.16.4': 56, '2.1': 57, '2.2': 58, '2.3.1': 59, '2.3.2': 60,
                  '2.3.3': 61, '2.3.4': 62, '2.4': 63, '2.5': 64, '2.6.1': 65, '2.6.2': 66, '2.7': 67, '3.1': 68,
                  '3.2': 69, '3.3': 70, '3.9': 71, '3.10': 72, '3.18.1': 73, '3.18.2': 74, '3.19': 75, '3.20.1': 76,
                  '3.24.1': 77, '3.27': 78, '3.28': 79, '4.1.1': 80, '4.1.2': 81, '4.1.3': 82, '4.1.4': 83, '4.1.5': 84,
                  '4.2.1': 85, '4.2.2': 86, '4.3': 87, '4.5.1': 88, '4.5.2': 89, '4.6.1': 90, '4.6.2': 91, '5.5': 92,
                  '5.6': 93, '5.7.1': 94, '5.7.2': 95, '5.11.1': 96, '5.12.1': 97, '5.15': 98, '5.16.1': 99, '5.16.2': 100}
    try:
        return row_labels[row_label]
    except:
        print(row_label)
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
