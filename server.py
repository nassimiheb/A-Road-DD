import sys, os
sys.path.append('./yolo')
sys.path.insert(4,'/usr/local/lib/python3.6/dist-packages')
print(sys.path)

from flask import Flask, request, jsonify
#from yolo.detect import *
import threading

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolo.models.experimental import attempt_load
from yolo.utils.datasets import LoadStreams, LoadImages
from yolo.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolo.utils.plots import plot_one_box
from yolo.utils.torch_utils import select_device, load_classifier, time_synchronized
from core import trips_management
import csv

app = Flask(__name__)
current_thread_id = 0

#Trigger to indecate control the detection script
exit_detect_event = threading.Event()
exit_detect_event.set()

# get all trips
TRIPS_FOLDER = './trips_data'
all_trips = trips_management.read_all_trips(TRIPS_FOLDER)
IMAGES_FOLDER = 'trip_images_save'

#initialize position global variables
coor_lat = 0
coor_long = 0

#model
model_wights = 'yolov5s.pt'

def return_error(msg, code):
    return jsonify({
                    'error' : True, 
                    'message' : msg
                }),code

def detect_async(opt, file_save, img_save_path = None, save_img=True, jetson = True):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))    
    if jetson:
        view_img = False

    if save_img:
        print(img_save_path)
        img_save_dir = Path(img_save_path, exist_ok = True)  # increment run
        img_save_dir.mkdir(parents = True, exist_ok = True)  # make dir
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, jetson = jetson)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    t0 = time.time()
    counter = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (int(cls) ,*xywh, coor_lat, coor_long)  # label format
                        with open(file_save, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(line)

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    
                if save_img:
                    cv2.imwrite(img_save_path + '/' + str(counter) + '.jpg', im0)
                    counter += 1

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
        
        if(exit_detect_event.is_set()):
            vid_cap.release()
            cv2.destroyAllWindows()
            break

    print(f'Done. ({time.time() - t0:.3f}s)')
        


@app.route('/coordinates', methods=['PUT'])
def result():
    global coor_lat
    global coor_long
    request_json = request.get_json()
    lat_ = request_json.get('lat', None)
    long_ = request_json.get('long', None)

    print(lat_,long_)
    if(lat_):
        coor_lat = lat_
    if(long_):
        coor_long = long_
    return jsonify({'message':'done'})


@app.route('/detect', methods=['GET'])
def detect():
    if(exit_detect_event.is_set()):
        id = request.args.get('id', default = None)
        save_img = request.args.get('save_img', default = False)
        jesave_img = request.args.get('jesave_img', default = True)
        if id and id in all_trips:
            exit_detect_event.clear()
            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str, default=model_wights, help='model.pt path(s)')
            parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
            parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
            parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='display results')
            parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented inference')
            parser.add_argument('--update', action='store_true', help='update all models')
            parser.add_argument('--project', default='runs/detect', help='save results to project/name')
            parser.add_argument('--name', default='exp', help='save results to project/name')
            parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

            opt = parser.parse_args()
            print(opt)
            check_requirements(file= 'yolo/requirements.txt')
            file_save = TRIPS_FOLDER + '/' + id +'.csv'
            img_save_path = 'static/' + IMAGES_FOLDER + '/' + id
            with torch.no_grad():
                if opt.update:  # update all models (to fix SourceChangeWarning)
                    for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                        thread_detect = threading.Thread(target=detect_async,args=(opt, file_save, img_save_path, save_img, jesave_img))
                        strip_optimizer(opt.weights)
                else:
                    thread_detect = threading.Thread(target=detect_async, args=(opt, file_save, img_save_path, save_img, jesave_img)) 

            thread_detect.start()
            return jsonify({'message':'done'})
        else: return return_error('The trip does not exist', 404)
    else: return return_error('There is a detection already', 404)

@app.route('/stop_detect', methods=['GET'])
def stop_detect():
    if(not exit_detect_event.is_set()):
        exit_detect_event.set()
        return jsonify({'message':'done'})
    else:
        return return_error('no detection executing', 404) 

@app.route('/trips', methods=['GET'])
def list_trips():
    return jsonify(all_trips)


@app.route('/trip/create', methods=['PUT'])
def create_trip():
    global all_trips
    request_json = request.get_json()
    name = request_json.get("name", None)
    if name:
        trips_management.create_trip(TRIPS_FOLDER, name)
        all_trips = trips_management.read_all_trips(TRIPS_FOLDER)
        return jsonify(
                {
                    'message' : 'Trip created'
                })
    return return_error('You have to provide a name', 404)

@app.route('/trip/delete', methods=['PUT'])
def delete_trip():
    global all_trips
    request_json = request.get_json()
    id = request_json.get("id", None)
    if id:
        if trips_management.delete_trip(TRIPS_FOLDER, id):
            all_trips = trips_management.read_all_trips(TRIPS_FOLDER)
            return jsonify(
                {
                    'message' : 'Trip deleted'
                })
        else:
            return return_error('The trip does not exist', 404)
    return return_error('You have to provide an id', 404)

@app.route('/trip/stats', methods=['GET'])
def trip_stats():
    id = request.args.get('id', default = None)
    if id:
        stats = trips_management.get_stats(TRIPS_FOLDER, id)
        if stats:
            return jsonify(stats)
        else:
            return return_error('The trip does not exist', 404)
    return return_error('You have to provide an id', 404)

@app.route('/trip/image', methods=['GET'])
def get_image():
    id = request.args.get('id', default = None)
    detection_number = request.args.get('detection_number', default = None)
    if id and detection_number:
        if id in all_trips:
            image = trips_management.get_image(IMAGES_FOLDER, id, detection_number, app)
            if image:
                return image
            else:
                return return_error('The specified image does not exist', 404)
        else:
            return return_error('The trip does not exist', 404)
    return return_error('You have to provide an id and a number', 404)

if __name__ == '__main__':
    app.debug = True
    app.run(host= '0.0.0.0')