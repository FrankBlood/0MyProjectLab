# -*-coding:utf8-*-
import tensorflow as tf
import numpy as np
import cv2, random
from io import BytesIO
from captcha.image import ImageCaptcha

# Create ocr dataset using captcha
def gen_rand():
    buf = ""
    for i in range(1):
        buf += str(random.randint(0,9))
    return buf

def get_label(buf):
    a = [int(x) for x in buf]
    return np.array(a)

def label_one_hot(label_all):
    label_batch = []
    for label in label_all:
        one = []
        for i in label:
            one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            one_hot[i] = 1
            one += one_hot
        label_batch.append(one)
    label_batch = np.array(label_batch)
    return label_batch

def gen_sample(captcha, width, height):
    num = gen_rand()
    img = captcha.generate(num)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, height))
    img = np.multiply(img, 1/255.0)
    #img = img.transpose(2, 0, 1)
    return (num, img)

class OCRIter(object):
    def __init__(self, count, batch_size, height, width):
        self.captcha = ImageCaptcha(fonts=['./times.ttf'])
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        
    def __iter__(self):
        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_sample(self.captcha, self.width, self.height)
                data.append(img)
                label.append(get_label(num))

            data_all = np.array(data)
            label_all = np.array(label)
            label_batch = label_one_hot(label_all)
            data_all = data_all.astype("float32")
            label_batch = label_batch.astype("float64")
            data_names = ['data']
            label_names = ['softmax_label']
            yield data_all, label_batch
            
    def reset(self):
        pass

# Define the Deepnet using ocrnet
def print_activations(t):
    print t.op.name, ' ', t.get_shape().as_list()

def inference(images, keep_prob1, keep_prob2):
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 3, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)# 64, 64, 192
        parameters += [kernel, biases]

    #conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 192, 160], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[160], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)# 64, 64, 160

    #conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 160, 96], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)# 64, 64, 96

    # pool1
    pool1 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    print_activations(pool1)# 32, 32, 96

    # dropout1    
    h_pool1_drop = tf.nn.dropout(pool1, keep_prob1)
    print 'dropout', h_pool1_drop

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(h_pool1_drop, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_activations(conv4)# 32, 32, 192
        parameters += [kernel, biases]

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 192, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)# 32, 32, 192

    # conv6
    with tf.name_scope('conv6') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 192, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv6)# 32, 32, 192

    # pool2
    pool2 = tf.nn.avg_pool(conv6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print_activations(pool2)# 16, 16, 192

    # dropout2    
    h_pool2_drop = tf.nn.dropout(pool2, keep_prob2)
    print 'dropout', h_pool2_drop

    #conv7
    with tf.name_scope('conv7') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(h_pool2_drop, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv7 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv7)# 16, 16, 192

    #conv8
    with tf.name_scope('conv8') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 192, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv8 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv8)# 16, 16, 192

    #conv9
    with tf.name_scope('conv9') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 192, 10], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv9 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv9)# 16, 16, 10

    #pool3
    pool3 = tf.nn.avg_pool(conv9, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool3')
    print_activations(pool3)# 16, 16, 10

    # fc1
    W_fc1 = weight_variable([16*16*10, 10])
    b_fc1 = bias_variable([10])
    h_pool3_flat = tf.reshape(pool3, [-1, 16*16*10])
    print 'h_pool5_flat', h_pool3_flat
    h_fc1 = tf.nn.softmax(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    print 'fc1', h_fc1

    return h_fc1

# Train
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def train():
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 64 ,64, 3])
    y_ = tf.placeholder("float", [None, 10])
    print '申请两个占位符'
    keep_prob1 = tf.placeholder("float")
    keep_prob2 = tf.placeholder("float")
    print '申请两个keep_prob'

    y_conv = inference(x, keep_prob1, keep_prob2)    

    saver = tf.train.Saver()
    cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
    print 'cross_entropy'
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    print 'train_step'
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    print 'correct_prediction'
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print 'accuracy'
    sess.run(tf.initialize_all_variables())
    print 'run'

    batch_size = 50
    print '生成数据'
    i = 0
    for (batch_data, label_batch) in OCRIter(100000, batch_size, 64, 64):
        if i%100 == 0:
            saver.save(sess, './ninocr', global_step=i)
            sum_accuracy = 0.0
            num = 0
            for (test_data, test_label) in OCRIter(10000, batch_size, 64, 64):
                test_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_label, keep_prob1: 1.0, keep_prob2: 1.0})
                sum_accuracy += test_accuracy
                num += 1
            print (i, sum_accuracy/num)
        train_step.run(feed_dict={x: batch_data, y_: label_batch, keep_prob1: 0.5, keep_prob2: 0.5})
        i = i + 1

if __name__ == '__main__':
    train()