from Benckmark.BenckmarkBase import BenckmarkBase


class AOR(BenckmarkBase):
    def get_iterator(self):
        # fixme 瞎写的
        gt, result = yield
        while True:
            gt_x, gt_y, gt_w, gt_h = gt
            result_x, result_y, result_w, result_h = result
            gt_center = (gt_x + gt_w / 2, gt_y + gt_h / 2)
            result_center = (result_x + result_w / 2, result_y + result_h / 2)
            gt, result = yield 1
