import mxnet as mx
from sklearn.neighbors import KNeighborsClassifier
from scipy import misc
# import sys
import os
# import argparse
#import tensorflow as tf
import numpy as np
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
# from time import sleep
# from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
import face_preprocess
import frozen
import multiprocessing

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        try:
            self.video = cv2.VideoCapture(0)
            self.detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(), num_worker=4, accurate_landmark = False, threshold=[0.6,0.7,0.7])
            sym, arg_params, aux_params = mx.model.load_checkpoint('model/model-r34-amf/model', 0)
            #arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
            self.model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names = None)
            #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
            self.model.bind(data_shapes=[('data', (1, 3, 112, 112))])
            self.model.set_params(arg_params, aux_params)
            self.neigh = KNeighborsClassifier(n_neighbors=1)
            self.X = np.load('base.npy')
            self.names = []
            with open('base.txt','r') as f:
                for line in f:
                    line = line.replace("\n","")
                    self.names.append(line)
            # print(self.names)
            y = np.arange(self.X.shape[0])
            # print(y)
            self.neigh.fit(self.X, y)
            os.system('cls')
            print('\n\n\n\n\n\t\t\t\tWHAT ARE YOUR COMMANDS?\n\n\n\n\n\t\t\tif you want to exit watching, please press \'q\'!')
        except:
            exit()
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    # def fun(self, model, img):
    #     input_blob = np.expand_dims(img, axis=0)
    #     data = mx.nd.array(input_blob)
    #     db = mx.io.DataBatch(data=(data,))
    #     self.model.forward(db, is_train=False)
    #     embedding = self.model.get_outputs()[0].asnumpy()
    #     embedding = sklearn.preprocessing.normalize(embedding)
    #     return embedding

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        results = self.detector.detect_face(image)
        if results is not None:
            total_boxes = results[0]
            points = results[1]
            draw = image
            b = total_boxes[0,0:4]
            p = points[0,:].reshape((2,5)).T
            nimg = face_preprocess.preprocess(image, b, p, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)  # ???
            img = np.transpose(nimg, (2,0,1))
            input_blob = np.expand_dims(img, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            self.model.forward(db, is_train=False)
            embedding = self.model.get_outputs()[0].asnumpy()
            em = sklearn.preprocessing.normalize(embedding)
            k = self.neigh.predict(em)[0]
            # mind = 2
            name = 'unknown'
            rgb = (255, 0, 0)
            if np.linalg.norm(self.X[k] - em[0]) < 1.24:
                name = self.names[k]
                rgb = (0, 255, 255)
            else:
                name = 'unknown'
            # for k in range(4):
            #     di = np.linalg.norm(self.X[k] - em[0])

            #     if di < 1.24:
            #         name = self.names[k]
            #         rgb = (0, 255, 255)
            #     # else:
            #         # name = 'unknown'
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), rgb)
            cv2.putText(draw, name, (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            image = draw
        cv2.imshow('capture', image)
        # cv2.waitKey(5)
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    vc = VideoCamera()
    while True:
        vc.get_frame()
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
