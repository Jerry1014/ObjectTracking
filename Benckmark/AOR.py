from Benckmark.BenckmarkBase import BenckmarkBase


class AOR(BenckmarkBase):
    def get_iterator(self):
        gt, result = yield
        while True:
            gt_x, gt_y, gt_w, gt_h = gt
            result_x, result_y, result_w, result_h = result
            start_x = max(gt_x, result_x)
            end_x = min(gt_x + gt_w, result_x + result_w)
            start_y = max(gt_y, result_y)
            end_y = min(gt_y + gt_h, result_y + result_h)
            radio = (end_x - start_x) * (end_y - start_y) if end_x > start_x and end_y > start_y else 0
            all_radio = gt_w * gt_h + result_w * result_h
            gt, result = yield radio / (all_radio - radio)
