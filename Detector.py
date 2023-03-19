# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 19:42:17 2023

@author: Nivellen
"""

import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(20)

class Detector:
    def init(self) :
        pass
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r')as f:
            self.classesList = f.read().splitlines()
            
        # Colors list
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
                                           
    def downloadmodel(self, modelURL):
        fileName= os.path.basename(modelURL)
        self.modelName= fileName[:fileName.index('.')]
        print(fileName)
        print(self.modelName)
        
        self.cacheDir = './pretrained_models'
        
        os.makedirs(self.cacheDir, exist_ok=True)
        
        get_file(fname=fileName, 
                 origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints",extract=True)
     
    def loadModel(self):
        print("loading model "+ self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        
        print("\nModel "+ self.modelName + " loaded successfully...\n")
        
        
    def createBoundigBox(self, image, threshold=0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]
        
        detections= self.model(inputTensor)
        
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()
        
        imH, imW, imC = image.shape
        
        # fix, too much boxs on image:
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, 
                                               iou_threshold=threshold, score_threshold=threshold)
      
        print(*bboxs, sep = ", ")
        
        if (len(bboxs) != 0):
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist()) 
                classConfidence = round(100*classScores[i])
                classIndex =classIndexes[i]
                
                classLabelText = self.classesList[classIndex]
                classColor= self.colorList[classIndex]
                
                print('classLabelText: ' + classLabelText)
                # print(*self.classesList, sep = ", ")
                print('classIndexes[i]: '+ str(classIndexes[i]))
                
                displayText = '{}: {}%'.format(classLabelText, classConfidence)
                
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
            
                # aditional line in box
                lineWidth = min(int((xmax - xmin) *0.2) , int((ymax - ymin)*  0.2))
                
                cv2.line(image, (xmin, ymin), (xmin + lineWidth ,ymin), classColor,  thickness=5)
                
        return image
                
                
    
    def predictImage(self, imagePath, threshold=0.5):
        image = cv2.imread(imagePath)
        
        bboxImage = self.createBoundigBox(image, threshold)
        
        cv2.imwrite(self.modelName +"_Result2"+ ".jpg", bboxImage)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
        
        