
# comment out below line to enable tensorflow outputs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
from pyvirtualdisplay import Display
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
#new imports for server
import sys
import glob
#FLAGS = flags.FLAGS
#from tensorflow.python.util import compat
import os
class FLAGZ: 
    framework = 'tf'
    weights= './checkpoints/yolov4-tiny-416'
    size = 416
    model = 'yolov4'
    video = None
    output = None
    output_format = 'XVID'
    iou = 0.45
    score = 0.50
    count = True
    dont_show=True
    info=False
    crop=False
    plate=False
    tiny=True
    
def main(video_path):
    disp = Display(visible=False)
    disp.start()
    FLAGS =FLAGZ()
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    frame_num = 0
    f = open('results.txt', 'w')
    f.close()
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            vid.release()
            session.close()
            break
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        # custom allowed classes (uncomment line below to allow detections for only people)
            
        # count objects found
        counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
        # loop through dict and print
        
        for key, value in counted_classes.items():
                f = open("results.txt", "a")
                f.write("{}:{}\n".format(key, value))
        f.close()
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    os.remove(video_path)
    t = open("cars.txt", "w")
    tr = open("trucks.txt","w")
    b = open("buses.txt","w")
    bi = open("bicycles.txt","w")
    f = open("results.txt", "r")
    for line in f:
        x = line.split(":")
        if x[0]=='car':
            t.write(line)
        elif x[0]=='truck':
            tr.write(line)
        elif x[0] == "bus":
            b.write(line)
        elif x[0] == "bicycle":
            bi.write(line)
    t.close()
    tr.close()
    b.close()
    bi.close()
    total_cars = count_vehicle("cars.txt")
    total_trucks = count_vehicle("trucks.txt")
    total_buses = count_vehicle("buses.txt")
    total_bicycles = count_vehicle("bicycles.txt")
    disp.stop()
    return {"carCount": total_cars, "truckCount": total_trucks, "busCount": total_buses, "bicycleCount": total_bicycles}

def count_vehicle(filename):
    file = open(filename, "r+")
    file = file.readlines()
    total = 0
    for i in range(0,len(file)):
        line = file[i]
        if i+1 < len(file):
            x = line.split(":")
            y = file[i+1].split(":")
            if int(y[1]) <  int(x[1]):
                total += int(x[1]) - int(y[1])
        else:
            x = line.split(":")
            total += int(x[1])
    del file
    return total

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
