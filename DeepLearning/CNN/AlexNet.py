# coding: utf-8
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

tfrecord = './../data/Dogs&Cats/dogcat.tfrecords'
learning_rate = 1e-4
BATCH_SIZE = 50
DISPLAY_STEP = 5
n_classes = 2
n_fc1 = 4096
n_fc2 = 2048
MAX_STEP = 500  # 一般大于10K


def get_batch(tfrecord=tfrecord):
    filename_queue = tf.train.string_input_producer([tfrecord])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [227, 227, 3])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=BATCH_SIZE,
                                                      min_after_dequeue=100,
                                                      num_threads=64,
                                                      capacity=200)
    return image_batch, label_batch


# def get_batch(image, label, image_W, image_H, batch_size, capacity):  
#     # image, label: 要生成batch的图像和标签list  
#     # image_W, image_H: 图片的宽高  
#     # batch_size: 每个batch有多少张图片  
#     # capacity: 队列容量  
#     # return: 图像和标签的batch  

#     # 将python.list类型转换成tf能够识别的格式  
#     image = tf.cast(image, tf.string)  
#     label = tf.cast(label, tf.int32)  

#     # 生成队列  
#     input_queue = tf.train.slice_input_producer([image, label])  

#     image_contents = tf.read_file(input_queue[0])  
#     label = input_queue[1]  
#     image = tf.image.decode_jpeg(image_contents, channels=3)  

#     # 统一图片大小  
#     # 视频方法  
#     # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)  
#     # 我的方法  
#     image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  
#     image = tf.cast(image, tf.float32)  
#     # image = tf.image.per_image_standardization(image)   # 标准化数据  
#     image_batch, label_batch = tf.train.batch([image, label],  
#                                               batch_size=batch_size,  
#                                               num_threads=64,   # 线程  
#                                               capacity=capacity)  

#     # label_batch = tf.reshape(label_batch, [batch_size])

#     return image_batch, label_batch  

def batch_norm(inputs, is_training, is_conv_out=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, 0.001)


def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.int32, [None, n_classes])

W_conv = {
    'conv1': tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.0001)),
    'conv2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01)),
    'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01)),
    'conv4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01)),
    'conv5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01)),
    'fc1': tf.Variable(tf.truncated_normal([13 * 13 * 256, n_fc1], stddev=0.1)),
    'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
    'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1))
}
b_conv = {
    'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96])),
    'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
    'conv3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
    'conv4': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
    'conv5': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
    'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1])),
    'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
    'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes]))
}

x_image = tf.reshape(x, [-1, 227, 227, 3])

# 卷积层 1
conv1 = tf.nn.conv2d(x_image, W_conv['conv1'], strides=[1, 4, 4, 1], padding='VALID')
conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
conv1 = batch_norm(conv1, True)
conv1 = tf.nn.relu(conv1)
# 池化层 1
pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
norm1 = tf.nn.lrn(pool1, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 卷积层 2
conv2 = tf.nn.conv2d(pool1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
conv2 = batch_norm(conv2, True)
conv2 = tf.nn.relu(conv2)
# 池化层 2
pool2 = tf.nn.avg_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# 卷积层3
conv3 = tf.nn.conv2d(pool2, W_conv['conv3'], strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
conv3 = batch_norm(conv3, True)
conv3 = tf.nn.relu(conv3)

# 卷积层4
conv4 = tf.nn.conv2d(conv3, W_conv['conv4'], strides=[1, 1, 1, 1], padding='SAME')
conv4 = tf.nn.bias_add(conv4, b_conv['conv4'])
conv4 = batch_norm(conv4, True)
conv4 = tf.nn.relu(conv4)

# 卷积层5
conv5 = tf.nn.conv2d(conv4, W_conv['conv5'], strides=[1, 1, 1, 1], padding='SAME')
conv5 = tf.nn.bias_add(conv5, b_conv['conv5'])
conv5 = batch_norm(conv5, True)
conv5 = tf.nn.relu(conv2)

# 池化层5
pool5 = tf.nn.avg_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
reshape = tf.reshape(pool5, [-1, 13 * 13 * 256])
fc1 = tf.add(tf.matmul(reshape, W_conv['fc1']), b_conv['fc1'])
fc1 = batch_norm(fc1, True, False)
fc1 = tf.nn.relu(fc1)

# 全连接层 2
fc2 = tf.add(tf.matmul(fc1, W_conv['fc2']), b_conv['fc2'])
fc2 = batch_norm(fc2, True, False)
fc2 = tf.nn.relu(fc2)
fc3 = tf.add(tf.matmul(fc2, W_conv['fc3']), b_conv['fc3'])

# 定义损失
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc3, labels=tf.argmax(y, 1)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
# 评估模型
correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

save_model = "./model/AlexNet.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(".//log", sess.graph)  # 输出日志的地方
    saver = tf.train.Saver()

    c = []
    start_time = time.time()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    step = 0
    for i in range(MAX_STEP):
        step = i
        image_batch, label_batch = get_batch()
        image, label = sess.run([image_batch, label_batch])

        labels = onehot(label)

        sess.run(optimizer, feed_dict={x: image, y: labels})
        loss_record = sess.run(loss, feed_dict={x: image, y: labels})
        print("now the loss is %f " % loss_record)

        c.append(loss_record)
        end_time = time.time()
        print('time: ', (end_time - start_time))
        start_time = end_time
        print("---------------%d step is finished-------------------" % i)
    print("Optimization Finished!")
    saver.save(sess, save_model)
    print("Model Save Finished!")

    coord.request_stop()
    coord.join(threads)
    plt.plot(c)
    plt.xlabel('Iter')
    plt.ylabel('loss')
    plt.title('lr=%f, bs=%d' % (learning_rate, BATCH_SIZE))
    plt.tight_layout()
    plt.savefig('cat_and_dog_AlexNet.jpg', dpi=200)


