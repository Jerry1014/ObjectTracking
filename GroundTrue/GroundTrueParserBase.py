class GroundTrueParserBase:
    @staticmethod
    def get_result_iterator(ground_true_filename):
        """
        返回gt的迭代器
        返回的的数据格式为tuple (x, y, w, h)
        """
        raise NotImplementedError()