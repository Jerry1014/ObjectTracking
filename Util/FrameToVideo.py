import os

import cv2

frame_dir = '../Resources/video/Walking2/img'
first_frame_sign = True
video_writer = None
if os.path.exists(frame_dir):
    _, _, file_list = next(os.walk(frame_dir))
    for filename in file_list:
        frame = cv2.imread(frame_dir + os.path.sep + filename)
        if frame is not None:
            if first_frame_sign:
                video_writer = cv2.VideoWriter(frame_dir.split('/')[-1] + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 24,
                                               frame.shape[-2::-1])
                first_frame_sign = False
            video_writer.write(frame)
else:
    print('视频帧路径错误')
video_writer.release()
