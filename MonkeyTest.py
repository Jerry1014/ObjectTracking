# -*- coding:utf-8 -*-
import unittest


class TestCaseForReadVideo(unittest.TestCase):
    from ReadVideo import ReadVideoFromFile, EndOfVideoError

    def test_for_read_video(self):
        from cv2.cv2 import imshow, waitKey, cvtColor, COLOR_RGB2BGR
        cap = TestCaseForReadVideo.ReadVideoFromFile()
        cap.init('./Resources/video/Browse1.mpg')
        print(cap.get_frame_total_num())
        while cap.is_open():
            try:
                frame = cap.get_one_frame()
            except TestCaseForReadVideo.EndOfVideoError:
                assert True
                break
            imshow('image', cvtColor(frame, COLOR_RGB2BGR))
            waitKey(0)
        cap.release_init()

    def test_for_read_pic_dir(self):
        from ReadVideo import ReadPicFromDir
        test_dir = 'Resources/video/Walking2/img'
        test = ReadPicFromDir()
        test.init(test_dir)
        from ReadVideo import EndOfVideoError
        while True:
            try:
                from cv2.cv2 import imshow, waitKey
                frame = test.get_one_frame()
                print(frame)
                imshow('image', frame)
                waitKey(0)
            except EndOfVideoError:
                assert True
                break


class TestCaseForInterface(unittest.TestCase):
    def test_for_interface(self):
        from Main import Start
        Start().run()
        assert True


if __name__ == '__main__':
    unittest.main()
