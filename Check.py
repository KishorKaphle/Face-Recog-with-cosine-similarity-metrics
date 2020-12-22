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
    a=sqrt(np.dot(v1, np.transpose(v1)))
    b=sqrt(np.dot(v2, np.transpose(v2)))
    if a==0 or b==0:
        return -1
    cos_dis=np.dot(v1, np.transpose(v2)) / (b * a)
    # print('cos:',cos_dis)
    
    return cos_dis


def calc_diff(arr1,arr2):
        # arr1,arr2 = np.expand_dims(arr1,axis=-1), np.expand_dims(arr2.transpose(1,0),axis=0)
        diff_np = np.subtract(arr1,arr2)
        dist_np = np.sum(np.power(diff_np,2))
        minimum_np = np.min(dist_np)
        # return minimum_np ,np.argmin(dist_np,axis=1)
        return minimum_np


# def calc_diff(b1,b2):
#     diff = b1-b2
#     dist = np.sum(np.power(diff, 2))
#     # print(dist)
#     return dist


onnx_path = "version-RFB-640.onnx"


# FaceEmb = np.load('/home/kishor/vsc/ilab/Face_recog/face_emb/FaceEmb.npy')
# FaceLabel = np.load('/home/kishor/vsc/ilab/Face_recog/face_emb/FaceLabel.npy')

dataset = np.load('/home/kishor/vsc/ilab/Face_recog/face_emb/Database.npy', allow_pickle='TRUE').item()
# print(dataset.keys())


fast_mtcnn = onnx_face(onnx_path)
# sess = ort.InferenceSession('arcface_r1001.onnx')
sess = ort.InferenceSession('ir_se.onnx')

# root = '/home/kishor/vsc/ilab/Face_recog/face_emb/testImage'

root = '/home/kishor/vsc/ilab/Face_recog/face_emb/dataset_for_testing'
coSim = []
EuDis = []

for root_Path in os.listdir(root):
    root_path = root + '/' + root_Path
    for path in os.listdir(root_path):
        # try:
            path = root_path + '/' + path
            frame = cv2.imread(path)
            frame = cv2.resize(frame,(640,480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            faces,prob = fast_mtcnn.inference(path)
            input_name = sess.get_inputs()[0].name
            for face in faces:
                Cpredicted = []
                Epredicted = []
                Chigh = []
                Elow = []
                face = cv2.resize(np.array(face),(112,112))
                Face = Image.fromarray(face)
                # cv2.imshow('123',face)
                # cv2.waitKey(0)
                face =test_transform(Face).numpy()
                face = np.expand_dims(face,0).astype(np.float32)  
                # for i in range(10):
                out = sess.run(None, {input_name: face})

                for key, value in dataset.items():
                    # print(key)
                    # print(value)
                    Cdistances = []
                    Edistances = []
                    for faceEmb in value:
                        # faceEmb = np.asarray(faceEmb)
                        # print('face Emb from database :', faceEmb.shape)
                        # print('face Emb from test :', out[0][0].shape)
                        Cdistances.append(similarity((out[0][0]), faceEmb[0]))
                        Edistances.append(calc_diff((out[0][0]), faceEmb[0]))
                        # print('cosine similarity: ', similarity(faceEmb[0], out[0][0])) 
                        # print('distances: ', len(distances))
                        Chighest = max(Cdistances)
                        Elowest = min(Edistances)
                        # print(highest)

                        if key == root_Path:
                            # print(f"same person {key} Cosine similarity {similarity((out[0][0]), faceEmb[0])} Eucludian distance {calc_diff((out[0][0]), faceEmb[0])}")
                            coSim.append(similarity((out[0][0]), faceEmb[0]))
                            EuDis.append(calc_diff((out[0][0]), faceEmb[0]))

                        if Chighest >=-0.3:
                            # print(key, highest)
                            Chigh.append((Chighest))
                            Cpredicted.append(key)
                        
                        if Elowest <=2.0:
                            Elow.append((Elowest))
                            Epredicted.append(key)

                Cbest = max(Chigh)
                Ebest = min(Elow)
                # print(predicted)
                Cindx = [i for i, j in enumerate(Chigh) if j ==Cbest]
                print(f"{root_Path} Cosine Matched with {Cbest}: ", Cpredicted[Cindx[0]])

                Eindx = [i for i, j in enumerate(Elow) if j ==Ebest]
                print(f"{root_Path} Eucluidian Matched with {Ebest}: ", Epredicted[Eindx[0]])



# plt.plot(coSim)
# plt.grid()

# plt.title(f"Cosine Similarity of same person: {min(coSim)} Max: {max(coSim)}")
# plt.savefig('Plot with position changes truth values of cosine similarity.png')

# plt.plot(EuDis)
# plt.grid()
# plt.title(f"Euclidian Distance of same person: {min(EuDis)} Max: {max(EuDis)}")
# plt.savefig('Plot with position changes truth values of Euclidian Distances.png')

        # except Exception as e:
        #     print('Error! ', e)
        #     pass
  

