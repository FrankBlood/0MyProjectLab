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

# parameters
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Define the Deepnet using ocrnet
def inference(x):

    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = conv2d(x, W_conv1) + b_conv1
    h_pool1 = tf.nn.relu(max_pool_2x2(h_conv1))
    print '第一层卷积', h_pool1

    W_conv2 = weight_variable([5, 5, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2	
    h_pool2 = tf.nn.relu(max_pool_2x2(h_conv2))
    print '第二层卷积', h_pool2

    W_conv3 = weight_variable([5, 5, 32, 32])
    b_conv3 = bias_variable([32])
    h_conv3 = conv2d(h_pool2, W_conv3) + b_conv3	
    h_pool3 = tf.nn.relu(max_pool_2x2(h_conv3))
    print '第三层卷积', h_pool3

    W_fc1 = weight_variable([8*8*32, 256])
    b_fc1 = bias_variable([256])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*32])
    h_fc1 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
    print 'flatten', h_fc1

    W_fc21 = weight_variable([256, 10])
    b_fc21 = bias_variable([10])
    h_fc21 = tf.nn.softmax(tf.matmul(h_fc1, W_fc21) + b_fc21)
    print '第一个数字', h_fc21

    return h_fc21

# Train
def train():
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 64 ,64, 3])
    y_ = tf.placeholder("float", [None, 10])
    print '申请两个占位符'

    y_conv = inference(x)

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
            saver.save(sess, './ocrNet', global_step=i)
            sum_accuracy = 0.0
            num = 0
            for (test_data, test_label) in OCRIter(10000, batch_size, 64, 64):
                test_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_label})
                sum_accuracy += test_accuracy
                num += 1
            print (i, sum_accuracy/num)
        train_step.run(feed_dict={x: batch_data, y_: label_batch})
        i = i + 1

if __name__ == '__main__':
    train()