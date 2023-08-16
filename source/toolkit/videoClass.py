import cv2
import numpy as np

#고용량 데이터이니만큼 최적화 필수 / plotmanager처럼 ram에 전부 올리지 말고 one by one
class VideoManager():
    def __init__(self, size=(40, 30)):
        self.size = size
        self.step = "init"
        self.data = None

    def reshape(self, content, size):
        content = np.reshape(content, size)
        content = content.astype(np.uint8)
        self.data = content
        self.step = "reshape"
        return self

    def makeVideoData(self):
        assert self.data.ndim > 2, "matrix dimension could be larger than 2"
        assert self.data is not None, "excute reshape method first"

        frame_list = []
        for frame in self.data :
            frame_list.append(frame)

        self.data = frame_list
        self.step = "makeVideoData"
        return self

    def saveVideo(self, filename="test.avi", size=(40, 30)):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoobject = cv2.VideoWriter(filename, fourcc, 10, self.size)
        # try:
        if self.step == "makeVideoData":
            for frame in self.data:
                videoobject.write(frame)
        # except Exception as e:
        #     print(e)
        videoobject.release()