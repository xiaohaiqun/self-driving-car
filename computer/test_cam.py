__author__ = 'zhengwang'

import numpy as np
import cv2
import time
import os
import cv2
cap = cv2.VideoCapture(0)

cap.set(3,320) #设置分辨率
cap.set(4,240)

class CollectTrainingData(object):
    
    def __init__(self, host, port, serial_port, input_size):

        #self.server_socket = socket.socket()
        #self.server_socket.bind((host, port))
        #self.server_socket.listen(0)

        # accept a single connection
        #self.connection = self.server_socket.accept()[0].makefile('rb')

        # connect to a seral port
        self.send_inst = True

        self.input_size = input_size

        # create labels
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1


    def collect(self):

        saved_frame = 0
        total_frame = 0

        X = np.empty((0, self.input_size))
        y = np.empty((0, 4))

        # stream video frames one by one
        try:
            stream_bytes = b' '
            frame = 1
            while self.send_inst:
                #stream_bytes += self.connection.read(1024)
                #first = stream_bytes.find(b'\xff\xd8')
                #last = stream_bytes.find(b'\xff\xd9')
                ret, frame = cap.read()

                if True:
                    image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    #jpg = stream_bytes[first:last + 2]
                    #stream_bytes = stream_bytes[last + 2:]
                    #image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    
                    # select lower half of the image
                    if image.any():
                        height, width = image.shape
                    else:
                        continue
                    roi = image[int(height/2):height, :]

                    cv2.imshow('image', image)

                    # reshape the roi image into a vector
                    temp_array = roi.reshape(1, int(height/2) * width).astype(np.float32)
                    
                    frame += 1
                    total_frame += 1

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


        finally:
            pass
            #self.connection.close()
            #self.server_socket.close()


if __name__ == '__main__':
    # host, port
    h, p = "127.0.0.1", 8000

    # serial port
    sp = "/dev/tty.usbmodem1421"

    # vector size, half of the image
    s = 120 * 320

    ctd = CollectTrainingData(h, p, sp, s)
    ctd.collect()
