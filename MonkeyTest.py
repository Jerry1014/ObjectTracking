# -*- coding:utf-8 -*-
import unittest


class TestCaseForReadVideo(unittest.TestCase):
    from ReadVideo import ReadVideoFromFile, EndOfVideoError

    def test_for_read_video(self):
        from cv2.cv2 import imshow, waitKey
        cap = TestCaseForReadVideo.ReadVideoFromFile()
        cap.open_video('./resources/video/因为我穷.mp4')
        while cap.is_open():
            try:
                frame = cap.get_one_frame()
                print(frame)
            except TestCaseForReadVideo.EndOfVideoError:
                assert True
                break
            imshow('image', frame)
            waitKey(0)
        cap.release_init()

    def test_for_read_pic_dir(self):
        from ReadVideo import ReadPicFromDir
        test_dir = './dataset/Walking2/img'
        test = ReadPicFromDir(test_dir)
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
