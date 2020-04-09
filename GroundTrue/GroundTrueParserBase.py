class GroundTrueParserBase:
    @staticmethod
    def get_result_list(ground_true_filename):
        """
        返回gt的列表
        返回的的数据格式为tuple (x, y, w, h)
        """
        raise NotImplementedError()