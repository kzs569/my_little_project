# 数据预处理
import cv2
import os
import traceback
import numpy as np
import tensorflow as tf

SOURCE_PATH = "./../data/Dogs&Cats/train/"
TARGET_PATH = "./../data/Dogs&Cats/preprocess/"


# 将原图进行统一的裁剪
def rebuild(source_dir=SOURCE_PATH, target_path=TARGET_PATH):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            filepath = os.path.join(root, file)
            print(filepath)
            try:
                image = cv2.imread(filepath)
                dim = (227, 227)
                resized = cv2.resize(image, dim)
                cv2.imwrite(target_path + file, resized)
            except:
                print(traceback.format_exc())
                print(filepath)
        cv2.waitKey(0)


# 读取相应图片的标签信息
def examplelist(path=TARGET_PATH):
    examplelist = []
    for root, sub_folders, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            # 0表示狗，1表示猫
            if 'cat' in file:
                examplelist.append([1, filepath])
            elif 'dog' in file:
                examplelist.append([0, filepath])
    np.random.shuffle(examplelist)
    return examplelist


def conver_to_tfrecord(examplelist, tfrecord='./../data/Dogs&Cats/dogcat.tfrecords'):
    writer = tf.python_io.TFRecordWriter(tfrecord)
    for i in range(len(examplelist)):
        try:
            image = cv2.imread(examplelist[i][1])
            image_raw = image.tostring()
            label = examplelist[i][0]
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print(e.message())
    writer.close()


def run(sourcepath, targetpath, tfrecord):
    rebuild(sourcepath=sourcepath,
            targetpath=targetpath)
    examplel = examplelist(targetpath=targetpath)
    conver_to_tfrecord(examplel, tfrecord)


if __name__ == '__main__':
    rebuild()
    conver_to_tfrecord(examplelist())
