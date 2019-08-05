'''
Change
- categories : {1 : "Blouse", 2 : "Tee", 3 : "Shorts", 4 : "Skirt", 5 : "Dress"
- limit the number of data : 10,000, 1900
- class weight : x
'''

import io
import os
import random
import re

import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('dataset_path', '', 'Path to DeepFashion project dataset with Anno, Eval and Img directories')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('categories', '', 'Define the level of categories; broad or fine')
flags.DEFINE_string('evaluation_status', '', 'train, val or test')
FLAGS = flags.FLAGS

LABEL_DICT = {1: "top", 2: "bottom", 3: "long"}

LABEL_CONVERT_DICT = {3 : 1, 18 : 2, 32 : 3, 33 : 4, 41 : 5}

def create_tf_example(example, path_root):

    # import image
    f_image = Image.open(path_root + example["image_name"])

    # get width and height of image
    width, height = f_image.size

    # read image as bytes string
    encoded_image_data = io.BytesIO()
    f_image.save(encoded_image_data, format='jpeg')
    encoded_image_data = encoded_image_data.getvalue()

    filename = example["image_name"]  # Filename of the image. Empty if image is not from file
    filename = filename.encode()

    image_format = 'jpeg'.encode()  # b'jpeg' or b'png'

    xmins = [example['x_1'] / width]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [example['x_2'] / width]  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [example['y_1'] / height]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [example['y_2'] / height]  # List of normalized bottom y coordinates in bounding box (1 per box)

    assert (xmins[0] >= 0.) and (xmaxs[0] < 1.01) and (ymins[0] >= 0.) and (ymaxs[0] < 1.01), \
        (example, width, height, xmins, xmaxs, ymins, ymaxs)

    if width < 50 or height < 50 \
        or (xmaxs[0] - xmins[0]) / (ymaxs[0] - ymins[0]) < 0.2 \
        or (xmaxs[0] - xmins[0]) / (ymaxs[0] - ymins[0]) > 5.:
        return None

    classes_text = [example['category_name'].encode()]  # List of string class name of bounding box (1 per box)
    classes = [LABEL_CONVERT_DICT[example['category_label']]]  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))
    return tf_example


def main(_):

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    dataset_path = FLAGS.dataset_path

    # Annotation file paths
    bbox_file = os.path.join(dataset_path, 'Anno/list_bbox.txt')
    cat_cloth_file = os.path.join(dataset_path, 'Anno/list_category_cloth.txt')
    cat_img_file = os.path.join(dataset_path, 'Anno/list_category_img.txt')
    stage_file = os.path.join(dataset_path, 'Eval/list_eval_partition.txt')

    # Read annotation files
    bbox_df = pd.read_csv(bbox_file, sep='\s+', skiprows=1)
    cat_cloth_df = pd.read_csv(cat_cloth_file, sep='\s+', skiprows=1)
    cat_img_df = pd.read_csv(cat_img_file, sep='\s+', skiprows=1)
    stage_df = pd.read_csv(stage_file, sep='\s+', skiprows=1)

    # Merge dfs
    cat_cloth_df["category_label"] = cat_cloth_df.index + 1
    cat_df = cat_img_df.merge(cat_cloth_df, how='left', on='category_label')
    examples_df = cat_df.merge(bbox_df, how='left', on='image_name')
    examples_df = examples_df.merge(stage_df, how='left', on='image_name')

    # Select train, val or test images
    examples_df = examples_df[examples_df["evaluation_status"] == FLAGS.evaluation_status]

    # filter category ###### edited
    filter_ctg = [3, 18, 32, 33, 41]
    examples_df = examples_df[examples_df['category_label'].apply(lambda x: True if x in filter_ctg else False)]

    # Shuffle
    examples_df = examples_df.sample(frac=1).reset_index(drop=True)

    ctg_num_dict = {ctg_label : 0 for ctg_label in filter_ctg}
    none_counter = 0
    for irow, example in examples_df.iterrows():

        ctg_num_dict[example.category_label] += 1
        if FLAGS.evaluation_status == "train": # limit the number of each category to maximum 10000
            if ctg_num_dict[example.category_label] > 10000 : continue
        if FLAGS.evaluation_status == "test": # limit the number of each category to maximum 1900
            if ctg_num_dict[example.category_label] > 1900 : continue

        try :
            tf_example = create_tf_example(example, path_root=os.path.join(dataset_path, 'Img/'))
        except AssertionError as e :
            print("assertion error at ", example["image_name"])
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())
        else:
            none_counter += 1
    print("Skipped %d images." % none_counter)

    writer.close()


if __name__ == '__main__':
    tf.app.run()
