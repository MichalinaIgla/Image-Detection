# -*- coding: utf-8-*-
"""
Created on Sun Mar 19 19:41:29 2023

@author: Nivellen
"""

from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
classFile = "coco.names"
imagePath="test/1.jpg"
threshold=0.5

print('\n------ RUN ME ------')

detector = Detector()
detector.readClasses(classFile)
detector.downloadmodel(modelURL)
detector.loadModel()
detector.predictImage(imagePath, threshold)