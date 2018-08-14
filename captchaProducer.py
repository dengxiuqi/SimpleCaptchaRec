import tensorflow as tf
from captcha.image import ImageCaptcha
import numpy as np  
import matplotlib.pyplot as plt  
from PIL import Image  
import random



number = ['0','1','2','3','4','5','6','7','8','9']  
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']  
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def random_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):
    # char_set: 验证码包含的字符集
    # captcha_size: 验证码的字符串长度
    captcha_text = []  
    for i in range(captcha_size):
        c = random.choice(char_set)  
        captcha_text.append(c)  
    return captcha_text  
   

def gen_captcha_text_and_image(width=160, height=60, char_set=number+alphabet+ALPHABET, captcha_size=4):
    image = ImageCaptcha(width=width, height=height)  # 默认图像大小是(160, 60, 3)
   
    captcha_text = random_captcha_text(char_set, captcha_size)
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    #image.write(captcha_text, captcha_text + '.jpg')

    captcha_image = Image.open(captcha)     # 打开图像
    captcha_image = np.array(captcha_image) # 保存成np格式
    return captcha_text, captcha_image
if __name__ == '__main__':
    text, image = gen_captcha_text_and_image()  
   
    f = plt.figure()  
    ax = f.add_subplot(111)  
    ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)  
    plt.imshow(image)
    print(image.shape)
   
    plt.show()  