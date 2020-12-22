import os
import time
from cv2 import cv2
import numpy as np

import onnxruntime as ort
import vision.utils.box_utils_numpy as box_utils
from facenet_pytorch import MTCNN
from PIL import Image
from math import sqrt
from torchvision import transforms as trans
import matplotlib.pyplot as plt


test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

class onnx_face:

    def __init__(self,onnx_path):
        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.threshold = 0.7
        self.iou_threshold = 0.3
        self.image_size = (640,480)

    def predict(self,width, height, confidences, boxes, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > self.threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                        iou_threshold=self.iou_threshold,
                                        top_k=top_k,
                                        )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def crop_image(self,orig_image,boxes):
        faces = []
        for box in boxes:
            # print(box)
            face = orig_image[box[1]:box[3],box[0]:box[2]]
            faces.append(face)
        return faces
    
    def inference(self,img_path):
        orig_image = cv2.imread(img_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (320, 240))
        image = cv2.resize(image, self.image_size)
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        confidences, boxes = self.ort_session.run(None, {self.input_name: image})
        boxes, labels, probs = self.predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes)
        if len(boxes) != 0:
            faces = self.crop_image(orig_image,boxes)
        else:
            faces = []
        return faces,probs

def similarity(v1, v2):
    
    a=sqrt(np.dot(v1, v1)) 
    b=sqrt(np.dot(v2, v2))
    if a==0 or b==0:
        return -1
    cos_dis=np.dot(v1, v2) / (b * a)
    # print('cos:',cos_dis)
    
    return cos_dis

def calc_diff(b1,b2):
    diff = b1-b2
    dist = np.sum(np.power(diff, 2))
    # print(dist)


onnx_path = "version-RFB-640.onnx"

embs = []
labels = []
Database = {'dummy': []}

# print(dataset)

fast_mtcnn = onnx_face(onnx_path)
# sess = ort.InferenceSession('arcface_r1001.onnx')
sess = ort.InferenceSession('ir_se.onnx')

root = '/home/kishor/vsc/ilab/Face_recog/face_emb/dataset_for_training'

for root_Path in os.listdir(root):
    root_path = root + '/' + root_Path
    for path in os.listdir(root_path):
        try:
            path = root_path + '/' + path
            frame = cv2.imread(path)
            frame = cv2.resize(frame,(640,480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            faces,prob = fast_mtcnn.inference(path)
            input_name = sess.get_inputs()[0].name
            for face in faces:
                face = cv2.resize(np.array(face),(112,112))
                face = Image.fromarray(face)
                face =test_transform(face).numpy()
                face = np.expand_dims(face,0).astype(np.float32)  
                a =time.time()
                for i in range(10):
                    out = sess.run(None, {input_name: face})
                key = root_Path
                if key in Database.keys():
                    print('True!')
                    Database[key].append([out[0]])
                else:
                    print('false!')
                    Database[key] = [out[0]]               
                # print(time.time()-a)
                # print(out[0])
                # embs.append(out[0])
                # labels.append(root_Path)

                # print(root_Path)
        except Exception as e:
            print('Error! ', e)
            pass
        # print(embs[0].shape)

# np.save('FaceEmb.npy', embs)
# np.save('FaceLabel.npy', labels)
np.save('Database.npy', Database)