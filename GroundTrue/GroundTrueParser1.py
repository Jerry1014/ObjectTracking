from GroundTrue.GroundTrueParserBase import GroundTrueParserBase


class GroundTrueParser1(GroundTrueParserBase):
    @staticmethod
    def get_result_list(ground_true_filename):
        with open(ground_true_filename) as f:
            return tuple(tuple(int(j) for j in i.split()) for i in f.readlines())
