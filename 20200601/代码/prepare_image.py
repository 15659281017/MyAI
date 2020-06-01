# encoding:utf-8
'''
对需要进行预测的图片进行处理并展示
'''
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing import image
import glob
import os
import random
import predict

# 返回一个图片的预测种类
def prepare(img_path, model):
    img = load_img(img_path, target_size=(512, 384))
    # 图像预处理
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    results = model.predict(x)
    return results

# 根据预测结果显示对应的文字label
classes_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 将预测种类返回对应字符串
def generate_result(results):
    for i in range(6):
        if(results[0][i] == 1):
            return classes_types[i]

# 随机显示15幅图像，并显示预测种类和实际种类
def prepare_image():
    model = predict.model()
    base_path = 'dataset/train'
    img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
    #print(len(img_list))

    plt.figure(figsize=(30, 15))
    for i, img_path in enumerate(random.sample(img_list, 15)):
        img = image.load_img(img_path)
        img = image.img_to_array(img, dtype=np.uint8)
        plt.subplot(3, 5, i + 1)

        res = prepare(img_path, model) # 返回每个图片的预测结果results
        str_p = generate_result(res) # 获取预测的类别

        # 例如图片路径为（有两种斜杠/、\）：dataset/validation\metal\metal230.jpg
        str_fold = img_path.split('/')  # 去掉前面的 dataset/
        str_fold2 = str_fold[1].split('\\', 2)  # 切分后面的 validation\metal\metal230.jpg
        str_fold3 = str_fold2[2].split('.')

        plt.title('pred:%s / truth:%s' % (str_p, str_fold3[0]))
        plt.imshow(img.squeeze())
    plt.savefig('results.png')
    plt.show('results.png')
