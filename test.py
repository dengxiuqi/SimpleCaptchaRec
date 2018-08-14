import captchaProducer
import train
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

char_map = {}
for i, char in enumerate(train.captcha_char):
    char_map[i] = char

def label2text(label):
    res = ''
    for item in label[0]:
        res += char_map[item]
    return res

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model_num = 19900
meta_path = 'models\\crack_capcha.model-%d.meta'%model_num
weight_path = 'models\\crack_capcha.model-%d'%model_num

saver = tf.train.import_meta_graph(meta_path)
saver.restore(sess, weight_path)
graph = tf.get_default_graph()

X = sess.graph.get_tensor_by_name("X_1:0")
y = sess.graph.get_tensor_by_name("y_1:0")
keep_prob = sess.graph.get_tensor_by_name("keep_prob_1:0")
res = graph.get_tensor_by_name("y_pred:0")

img, label = train.get_next_batch(1, testing=True)
res = sess.run(res, feed_dict={X:img,keep_prob: 1.0})
print("prediction:", label2text(res))

