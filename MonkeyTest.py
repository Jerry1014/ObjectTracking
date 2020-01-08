# -*- coding:utf-8 -*-
import unittest


class TestCaseForReadVideo(unittest.TestCase):
    from ReadVideo import ReadVideoFromFile, OpenVideoError, EndOfVideo

    def test_for_open_video(self):
        cap = TestCaseForReadVideo.ReadVideoFromFile()
        cap.open_video('./resources/video/因为我穷.mp4')
        assert cap.is_open()
        try:
            cap.open_video('./resources/不存在.mp4')
        except TestCaseForReadVideo.OpenVideoError:
            assert True
        cap.release_init()

    def test_for_read_video(self):
        from cv2.cv2 import imshow, waitKey
        cap = TestCaseForReadVideo.ReadVideoFromFile()
        cap.open_video('./resources/video/因为我穷.mp4')
        while cap.is_open():
            try:
                frame = cap.get_one_frame()
            except TestCaseForReadVideo.EndOfVideo:
                assert True
                break
            imshow('image', frame)
            waitKey(0)
        cap.release_init()


if __name__ == '__main__':
    unittest.main()
