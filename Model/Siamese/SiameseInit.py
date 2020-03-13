"""
Siamese网络的实现
"""
import tensorflow as tf

class Siamese(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 疑问1 是否为2d卷积
        # 疑问2 是否为最大值池化，且维度如何
        # 疑问3 默认参数是否存在问题
        # 3.1 conv padding='valid' use_bias=True kernel_initializer='glorot_uniform' bias_initializer='zeros'
        # 3.2 pool padding='valid'

        self.conv1 = Conv2D(96, 11, 2)
        self.pool1 = MaxPooling2D((3, 3), 2)
        self.conv2 = Conv2D(256, 5)
        self.pool2 = MaxPooling2D((3, 3), 2)
        self.conv3 = Conv2D(192, 3)
        self.conv4 = Conv2D(192, 3)
        self.conv5 = Conv2D(128, 3)

        self.conv6 = Conv2D(1, 6)

    def call(self, inputs, training=None, mask=None):
        exemplar_input = inputs[0]
        search_input = inputs[1]

        exemplar_output = self.train_1(exemplar_input)
        search_output = self.train_1(search_input)
        self.conv6.set_weights(exemplar_output)
        return self.conv6(search_output)

    def train_1(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
