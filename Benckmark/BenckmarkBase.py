"""
目标跟踪评价基类
"""


class BenckmarkBase:
    def get_iterator(self):
        """
        返回一个迭代器，通过send将元组(基准(x,y,w,h)，模型结果)送入，返回评价分数
        """
        raise NotImplementedError()
