"""
Siamese网络的实现
"""
import tensorflow as tf


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=length))


def conv2d(x, w, strides):
    return tf.nn.conv2d(x, w, strides=strides, padding='VALID')


def max_pool_2x2(inputx):
    return tf.nn.max_pool(inputx, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')


exemplar_image = tf.placeholder(tf.float32, shape=[1, 127, 127, 3])
search_image = tf.placeholder(tf.float32, shape=[1, 255, 255, 3])

# Conv 1
parameter_exemplar_conv1 = {'weight': new_weights([11, 11, 3, 96]),
                            'biases': new_biases([96])}
parameter_search_conv1 = {'weight': new_weights([11, 11, 3, 96]),
                          'biases': new_biases([96])}

with tf.name_scope('conv1_exemplar'):
    conv1_exemplar = conv2d(exemplar_image, parameter_exemplar_conv1['weight'], [2, 2]) + parameter_exemplar_conv1[
        'biases']
with tf.name_scope('conv1_search'):
    conv1_search = conv2d(search_image, parameter_search_conv1['weight'], [2, 2]) + parameter_search_conv1['biases']

# Pool 1
pool1_exemplar = max_pool_2x2(conv1_exemplar)
pool1_search = max_pool_2x2(conv1_search)

# Conv 2
parameter_exemplar_conv2 = {'weight': new_weights([5, 5, 96, 256]),
                            'biases': new_biases([256])}
parameter_search_conv2 = {'weight': new_weights([5, 5, 96, 256]),
                          'biases': new_biases([256])}

with tf.name_scope('conv2_exemplar'):
    conv2_exemplar = conv2d(pool1_exemplar, parameter_exemplar_conv2['weight'], [1, 1]) + parameter_exemplar_conv2[
        'biases']
with tf.name_scope('conv2_search'):
    conv2_search = conv2d(pool1_search, parameter_search_conv2['weight'], [1, 1]) + parameter_search_conv2['biases']

# Pool 2
pool2_exemplar = max_pool_2x2(conv2_exemplar)
pool2_search = max_pool_2x2(conv2_search)

# Conv 3
parameter_exemplar_conv3 = {'weight': new_weights([3, 3, 256, 192]),
                            'biases': new_biases([192])}
parameter_search_conv3 = {'weight': new_weights([3, 3, 256, 192]),
                          'biases': new_biases([192])}

with tf.name_scope('conv3_exemplar'):
    conv3_exemplar = conv2d(pool2_exemplar, parameter_exemplar_conv3['weight'], [1, 1]) + parameter_exemplar_conv3[
        'biases']
with tf.name_scope('conv3_search'):
    conv3_search = conv2d(pool2_search, parameter_search_conv3['weight'], [1, 1]) + parameter_search_conv3['biases']

# Conv 4
parameter_exemplar_conv4 = {'weight': new_weights([3, 3, 192, 192]),
                            'biases': new_biases([192])}
parameter_search_conv4 = {'weight': new_weights([3, 3, 192, 192]),
                          'biases': new_biases([192])}

with tf.name_scope('conv4_exemplar'):
    conv4_exemplar = conv2d(conv3_exemplar, parameter_exemplar_conv4['weight'], [1, 1]) + parameter_exemplar_conv4[
        'biases']
with tf.name_scope('conv4_search'):
    conv4_search = conv2d(conv3_search, parameter_search_conv4['weight'], [1, 1]) + parameter_search_conv4['biases']

# Conv 5
parameter_exemplar_conv5 = {'weight': new_weights([3, 3, 192, 128]),
                            'biases': new_biases([128])}
parameter_search_conv5 = {'weight': new_weights([3, 3, 192, 128]),
                          'biases': new_biases([128])}

with tf.name_scope('conv5_exemplar'):
    conv5_exemplar = conv2d(conv4_exemplar, parameter_exemplar_conv5['weight'], [1, 1]) + parameter_exemplar_conv5[
        'biases']
with tf.name_scope('conv5_search'):
    conv5_search = conv2d(conv4_search, parameter_search_conv5['weight'], [1, 1]) + parameter_search_conv5['biases']

# Conv6
parameter_conv6 = {'weight': tf.transpose(conv5_exemplar, perm=[1, 2, 3, 0]),
                   'biases': new_biases([128])}

with tf.name_scope('score_map'):
    score_map = conv2d(conv5_search, parameter_conv6['weight'], [1, 1]) + parameter_conv6['biases']

with tf.name_scope('train'):
    y = tf.placeholder(tf.float32, shape=[1, 22, 22, 1])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=score_map))
    optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

merged_data = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter('./log/', graph=sess.graph)
