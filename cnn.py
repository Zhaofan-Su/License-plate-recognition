import sys
import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="CNN Net")
parser.add_argument('-t', '--type', type=int,
                    help='0 for char CNN Net, 1 for license CNN Net')
parser.add_argument('-bz', '--batch_size', default=100,
                    type=int, help='The batch size of the cnn net.')
parser.add_argument('-lr', '--learn_rate', default=0.001,
                    type=float, help='The learning rate of the cnn net.')
parser.add_argument('-dr', '--data_dir', default='dataset/char-train', type=str,
                    help='The path of the  train dataset. char-train and license-train')
parser.add_argument('-tr', '--test_dir', default='dataset/char-test',
                    type=str, help='The path of the test dataset. char-test and license-test')
parser.add_argument('-mr', '--model_dir', default='char-models/char-model',
                    type=str, help='The path of the models.')
parser.add_argument('-m', '--model', type=str, help='The model to use.')
parser.add_argument('-f', '--function', default=0, type=int,
                    help='0 for train the net, 1 for test the net.')

args = parser.parse_args()


number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphebet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
province = ['zh-chongqing', 'zh-chuan', 'zh-e', 'zh-gan', 'zh-gansu', 'zh-gui', 'zh-guilin',
            'zh-hei', 'zh-hu', 'zh-ji', 'zh-jilin', 'zh-jin', 'zh-jing', 'zh-liao', 'zh-lu', 'zh-meng', 'zh-min', 'zh-ning',
            'zh-qing', 'zh-qiong', 'zh-shan', 'zh-shanxi', 'zh-su', 'zh-wan', 'zh-xiang', 'zh-xin', 'zh-yu', 'zh-yue', 'zh-yun', 'zh-zang', 'zh-zhe']


class CNN_NET:
    def __init__(self, net_type=0, batch_size=100, learn_rate=0.001):
        # 0 表示构建字符识别的CNN网络
        # 1 表示构建车牌提取分割的CNN网络
        if net_type == 0:
            self.type = 0
            self.channel = 1    # 图像通道数为1
            self.img_w = 20
            self.img_h = 20
            self.dataset = number + alphebet + province
            self.y_size = len(self.dataset)   # 10个数字，26个英文字母，31个省份简称
        elif net_type == 1:
            self.type = 1
            self.channel = 3    # 彩色图像，通道数为3
            self.img_w = 136
            self.img_h = 36
            self.y_size = 2     # 图像中是否有车牌两种情况
        else:
            raise ValueError("请选择模型！")

        self.batch_size = batch_size
        self.learning_rate = learn_rate
        if self.type == 0:
            self.x_place = tf.placeholder(dtype=tf.float32, shape=[
                None, self.img_h, self.img_w], name="x_place")
        elif self.type == 1:
            self.x_place = tf.placeholder(dtype=tf.float32, shape=[
                None, self.img_h, self.img_w, 3], name="x_place")
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[
                                      None, self.y_size], name="y_place")
        self.keep_place = tf.placeholder(dtype=tf.float32, name="keep_place")

    def construct(self):
        # shape: 图像宽、高、通道数
        x_input = tf.reshape(
            self.x_place, shape=[-1, self.img_h, self.img_w, self.channel])

        cw1 = tf.Variable(tf.random_normal(
            shape=[3, 3, self.channel, 32], stddev=0.01), dtype=tf.float32)
        cb1 = tf.Variable(tf.random_normal(shape=[32]), dtype=tf.float32)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
            x_input, filter=cw1, strides=[1, 1, 1, 1], padding="SAME"), cb1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding="SAME")
        conv1 = tf.nn.dropout(conv1, self.keep_place)

        cw2 = tf.Variable(tf.random_normal(
            shape=[3, 3, 32, 64], stddev=0.01), dtype=tf.float32)
        cb2 = tf.Variable(tf.random_normal(shape=[64]), dtype=tf.float32)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
            conv1, filter=cw2, strides=[1, 1, 1, 1], padding="SAME"), cb2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding="SAME")
        conv2 = tf.nn.dropout(conv2, self.keep_place)

        cw3 = tf.Variable(tf.random_normal(
            shape=[3, 3, 64, 128], stddev=0.01), dtype=tf.float32)
        cb3 = tf.Variable(tf.random_normal(shape=[128]), dtype=tf.float32)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
            conv2, filter=cw3, strides=[1, 1, 1, 1], padding="SAME"), cb3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding="SAME")
        conv3 = tf.nn.dropout(conv3, self.keep_place)

        if self.type == 0:
            conv_out = tf.reshape(conv3, shape=[-1, 3 * 3 * 128])

            fw1 = tf.Variable(tf.random_normal(
                shape=[3 * 3 * 128, 1024], stddev=0.01), dtype=tf.float32)
        elif self.type == 1:
            conv_out = tf.reshape(conv3, shape=[-1, 17 * 5 * 128])
            fw1 = tf.Variable(tf.random_normal(
                shape=[17 * 5 * 128, 1024], stddev=0.01), dtype=tf.float32)

        fb1 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully1 = tf.nn.relu(tf.add(tf.matmul(conv_out, fw1), fb1))
        fully1 = tf.nn.dropout(fully1, self.keep_place)

        fw2 = tf.Variable(tf.random_normal(
            shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fb2 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully2 = tf.nn.relu(tf.add(tf.matmul(fully1, fw2), fb2))
        fully2 = tf.nn.dropout(fully2, self.keep_place)

        fw3 = tf.Variable(tf.random_normal(
            shape=[1024, self.y_size], stddev=0.01), dtype=tf.float32)
        fb3 = tf.Variable(tf.random_normal(
            shape=[self.y_size]), dtype=tf.float32)
        fully3 = tf.add(tf.matmul(fully2, fw3), fb3, name="out_put")

        return fully3

    def train(self, data_path, save_model_path):
        print("载入训练数据……")
        X, y = self.load_data(data_path)
        print("训练数据载入完成")

        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=0)

        print("模型初始化……")
        out_put = self.construct()
        predicts = tf.nn.softmax(out_put)
        predicts = tf.argmax(predicts, axis=1)
        actual_y = tf.argmax(self.y_place, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicts, actual_y), dtype=tf.float32))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=out_put, labels=self.y_place))
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = opt.minimize(cost)
        print("模型初始化已完成！")

        print("模型训练开始……")
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            saver = tf.train.Saver()
            while True:
                train_index = np.random.choice(
                    len(train_X), self.batch_size, replace=False)
                train_randx = train_X[train_index]
                train_randy = train_y[train_index]
                _, loss = sess.run([train_step, cost], feed_dict={
                                   self.x_place: train_randx, self.y_place: train_randy, self.keep_place: 0.75})
                step += 1

                if step % 10 == 0:
                    test_index = np.random.choice(
                        len(test_X), self.batch_size, replace=False)
                    test_randx = test_X[test_index]
                    test_randy = test_y[test_index]
                    acc = sess.run(accuracy, feed_dict={
                                   self.x_place: test_randx, self.y_place: test_randy, self.keep_place: 1.0})

                    print("Step - %d: Loss= %f ; Accuracy= %f" %
                          (step, loss, acc))

                    if step % 100 == 0:
                        saver.save(sess, save_model_path, global_step=step)
                    if acc > 0.99 and step > 500:
                        saver.save(sess, save_model_path, global_step=step)
                        print("模型训练已完成！")
                        break

    def test(self, images, image_names, model_path):
        print("载入模型，测试开始……")
        text_list = []
        out_put = self.construct()
        predicts = tf.nn.softmax(out_put)
        probabilities = tf.reduce_max(predicts, reduction_indices=[1])
        predicts = tf.argmax(predicts, axis=1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            preds, probs = sess.run([predicts, probabilities], feed_dict={
                                    self.x_place: images, self.keep_place: 1.0})
            for i in range(len(preds)):
                pred = preds[i].astype(int)
                if self.type == 0:
                    text_list.append(
                        image_names[i] + ": " + self.dataset[pred])
                elif self.type == 1:
                    if pred == 1:
                        text_list.append(
                            image_names[i] + ": " + 'license-' + str(probs[i]))
                    else:
                        text_list.append(
                            image_names[i] + ": " + 'no-' + str(probs[i]))
            print("测试已完成！\n输出结果：")
            return text_list

    def list_all_files(self, path):
        files = []
        _list = os.listdir(path)
        for i in range(len(_list)):
            el = os.path.join(path, _list[i])
            if os.path.isdir(el):
                files.extend(self.list_all_files(el))
            elif os.path.isfile(el):
                files.extend([el])
        return files

    def load_data(self, data_path):
        X = []
        y = []
        if not os.path.exists(data_path):
            raise ValueError("没有找到数据集")
        files = self.list_all_files(data_path)
        if self.type == 0:
            for file in files:
                img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
                if img.ndim == 3:
                    continue
                resize_image = cv2.resize(img, (self.img_w, self.img_h))
                X.append(resize_image)
                # 获取图片文件全目录
                path = os.path.dirname(file)
                # 获取图片文件上一级目录名
                dir_name = os.path.split(path)[-1]
                vector_y = [0 for i in range(len(self.dataset))]
                index_y = self.dataset.index(dir_name)
                vector_y[index_y] = 1
                y.append(vector_y)
        elif self.type == 1:
            labels = [os.path.split(os.path.dirname(file))[-1]
                      for file in files]
            for i, file in enumerate(files):
                img = cv2.imread(file)
                if img.ndim != 3:
                    continue
                resize_image = cv2.resize(img, (self.img_w, self.img_h))
                X.append(resize_image)
                y.append([0, 1] if labels[i] == "has" else [1, 0])

        X = np.array(X)
        y = np.array(y).reshape(-1, self.y_size)
        return X, y

    def init_testData(self, test_dir):
        print("载入测试数据……")
        test_X = []
        image_names = []
        if not os.path.exists(test_dir):
            raise ValueError("没有找到文件夹")
        files = self.list_all_files(test_dir)
        for file in files:
            img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if (self.type == 0 and img.ndim == 3) or (self.type == 1 and img.ndim != 3):
                continue
            resize_image = cv2.resize(img, (self.img_w, self.img_h))
            test_X.append(resize_image)
            image_names.append(file.split('\\')[-1])
        test_X = np.array(test_X)
        print("测试数据载入完成！")
        return test_X, image_names


if __name__ == '__main__':
    current_path = sys.path[0]
    data_path = os.path.join(current_path, args.data_dir)
    test_path = os.path.join(current_path, args.test_dir)
    model_path = os.path.join(current_path, args.model_dir)
    print(test_path)

    if not os.path.exists(model_path):
        os.mkdir(args.model_dir)

    net_type = args.type
    batch_size = args.batch_size
    learn_rate = args.learn_rate
    function = args.function
    model = args.model
    if net_type == None:
        raise ValueError("请选择模型类型！")

    if function == 0:
        if net_type == 1 and 'dataset/char-train' in data_path:
            data_path = 'dataset/license-train'
            data_path = os.path.join(current_path, data_path)
        if net_type == 1 and 'char-models/char-model' in model_path:
            model_path = 'license-models/license-model'
            model_path = os.path.join(current_path, model_path)
        net = CNN_NET(net_type, batch_size, learn_rate)
        net.train(data_path, model_path)
    elif function == 1:
        net = CNN_NET(net_type, batch_size, learn_rate)
        test_X, image_names = net.init_testData(test_path)
        if model == None:
            raise ValueError("请选择已训练模型")
        trained_model_path = os.path.join(current_path, model)
        print(trained_model_path)
        text = net.test(test_X, image_names, model)
        print(text)
