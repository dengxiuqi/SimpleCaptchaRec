import captchaProducer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# global param
image_width = 160
image_height = 64
captcha_char = captchaProducer.number+captchaProducer.alphabet+captchaProducer.ALPHABET  # 只有数字
captcha_size = 4
captcha_varieties = len(captcha_char)
char_map = {}
for i, char in enumerate(captcha_char):
    char_map[char] = i

# training param
learn_rate = 1e-3
batch_size = 64
W_size = 3
pool_size = 2
conv1_num = 32
conv2_num = 32
conv3_num = 64
fc_num = 1024

# place_holders
X = tf.placeholder(tf.float32, [None, image_height, image_width], name='X')
y = tf.placeholder(tf.float32, [None, captcha_size, captcha_varieties], name='y')
keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')


def text2label(text):
    label = np.zeros(shape=(captcha_size, captcha_varieties), dtype=np.float32)
    for col, char in enumerate(text):
        label[col, char_map[char]] = 1
    return label

def convert2gray(img):
    if len(img.shape) > 2:
        # gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

def get_next_batch(batch_size, testing=None):
    if not testing:
        X_batch, y_batch = [], []
        for i in range(batch_size):
            text, image = captchaProducer.gen_captcha_text_and_image(image_width, image_height, captcha_char, captcha_size)
            label = text2label(text)
            img = convert2gray(image)
            X_batch.append(img)
            y_batch.append(label)
        X_batch, y_batch = (np.array(X_batch)-128)/128, np.array(y_batch)
        return X_batch, y_batch
    else:
        X_batch, y_batch = [], []
        text, image = captchaProducer.gen_captcha_text_and_image(image_width, image_height, captcha_char,
                                                                 captcha_size)
        label = text2label(text)
        img = convert2gray(image)
        X_batch.append(img)
        y_batch.append(label)
        X_batch, y_batch = (np.array(X_batch) - 128) / 128, np.array(y_batch)
        print('label:', text)
        plt.figure()
        plt.imshow(image)
        plt.show()
        return X_batch, y_batch

def cnn_structure(input):
    input = tf.expand_dims(input, -1)
    with tf.name_scope('conv1'):
        W1 = tf.Variable(tf.truncated_normal([W_size, W_size, 1, conv1_num], stddev=0.05), dtype=tf.float32, name='W')
        b1 = tf.Variable(tf.constant(0.05, shape=[conv1_num]), dtype=tf.float32, name='b')
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, W1, strides=[1, 1, 1, 1], padding='SAME'), b1), name='conv')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME', name='pool')
        dropout1 = tf.nn.dropout(pool1, keep_prob)

    with tf.name_scope('conv2'):
        W2 = tf.Variable(tf.truncated_normal([W_size, W_size, conv1_num, conv2_num], stddev=0.05), dtype=tf.float32, name='W')
        b2 = tf.Variable(tf.constant(0.05, shape=[conv2_num]), dtype=tf.float32, name='b')
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dropout1, W2, strides=[1, 1, 1, 1], padding='SAME'), b2),name='conv')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME',name='pool')
        dropout2 = tf.nn.dropout(pool2, keep_prob)

    with tf.name_scope('conv3'):
        W3 = tf.Variable(tf.truncated_normal([W_size, W_size, conv2_num, conv3_num], stddev=0.05), dtype=tf.float32, name='W')
        b3 = tf.Variable(tf.constant(0.05, shape=[conv3_num]), dtype=tf.float32, name='b')
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dropout2, W3, strides=[1, 1, 1, 1], padding='SAME'), b3),name='conv')
        pool3 = tf.nn.max_pool(conv3, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME',name='pool')
        dropout3 = tf.nn.dropout(pool3, keep_prob)

    with tf.name_scope('fc'):
        pixel_total = int(conv3_num * image_width/(pool_size ** 3) * image_height/(pool_size ** 3))
        dropout3_flat = tf.reshape(dropout3, [-1, pixel_total])
        W_fc = tf.Variable(tf.truncated_normal([pixel_total, fc_num], stddev=0.05), dtype=tf.float32, name='W')
        b_fc = tf.Variable(tf.constant(0.05, shape=[fc_num]), dtype=tf.float32, name='b')
        fc = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(dropout3_flat, W_fc, b_fc)), keep_prob)

    with tf.name_scope('output'):
        W_output = tf.Variable(
            tf.truncated_normal([fc_num, captcha_varieties * captcha_size], stddev=0.05), dtype=tf.float32, name='W')
        b_output = tf.Variable(tf.constant(0.05, shape=[captcha_varieties * captcha_size]), dtype=tf.float32, name='b')
        result = tf.reshape(tf.nn.xw_plus_b(fc, W_output, b_output), [-1, captcha_size, captcha_varieties], name='result')

    return result

def train():
    output = cnn_structure(X)
    # loss = tf.abs(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output, logits=y)))
    loss = tf.reduce_sum(tf.square(output-y))
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    y_pred = tf.argmax(output, dimension=-1, name='y_pred')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, tf.argmax(y, dimension=-1)), dtype=tf.float32), name='accuracy')
    saver = tf.train.Saver(max_to_keep=2)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        # train
        X_batch, y_batch = get_next_batch(batch_size)
        _, _loss, _acc = sess.run([optimizer, loss, accuracy], feed_dict={X:X_batch, y:y_batch, keep_prob:0.75})
        if step % 100 == 0:
            # test
            X_batch, y_batch = get_next_batch(batch_size)
            _loss, _acc = sess.run([loss, accuracy], feed_dict={X: X_batch, y: y_batch, keep_prob: 1.})
            print('step:', step, 'accuracy:', _acc, 'loss', _loss)
            saver.save(sess, './models/crack_capcha.model', global_step=step)
        if _acc >= 0.95:
            break

if __name__ == '__main__':
    train()