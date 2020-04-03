from GroundTrue.GroundTrueParserBase import GroundTrueParserBase


class GroundTrueParser1(GroundTrueParserBase):
    @staticmethod
    def get_result_iterator(ground_true_filename):
        with open(ground_true_filename) as f:
            a_line = f.readline()
            while a_line:
                yield (int(i) for i in a_line.split())
                a_line = f.readline()
