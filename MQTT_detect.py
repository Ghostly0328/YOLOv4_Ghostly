import argparse
import os
import platform
import shutil
import time

import cv2
import rsa
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

#MQTT
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

#MySQL
import pymysql

import numpy as np

def on_connect(client, userdata, flags, rc):
    if rc==0:
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)
    # Scbscribe Path
    client.subscribe(MQTT_PATH)

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def on_message(client, userdata, msg):      
    # Start Detect------------------------------------------------
    t0 = time.time()

    ShopID = "None"
    UUID = msg.topic.split('/')[-1]
    print('---UUID:%s---ShopID:%s---' % (UUID,ShopID))

    # Data Transform
    im0s = msg.payload    # Bytes
    im0s = np.asarray(im0s)    # Bytes2Np
    im0s = np.frombuffer(im0s, dtype=np.uint8)  #Buf
    im0s = cv2.imdecode(im0s, cv2.IMREAD_COLOR) #Np_Decode
  
    # Padded resize
    img = letterbox(im0s, new_shape=640, auto_size=64)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.cuda() if half else img  # uint8 to fp16/32 #Add Cuda()
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
    
    price = 0
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 ='', im0s
        #s += '%gx%g ' % img.shape[2:]  # print string

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%s %g, ' % (names[int(c)].capitalize(), n)  # add to string

                for index in ProductName[:,1:3]:
                    if names[int(c)].capitalize() == index[0]:
                        price += int(index[1])*n
                # if names[int(c)] == "person":
                #     price += 10
                # elif names[int(c)] == "bus":
                #     price += 2
            s += '%d' % (price)

            # Write results
            for *xyxy, conf, cls in det:
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

    # if s != '':     #有東西才做傳送
            
    print('%s' % (s))
    # Sent picture
    _,encoded_image = cv2.imencode(".jpg",im0)    # 对数组的图片格式进行编码
    img_bytes = encoded_image.tobytes() # 将数组转为bytes
    byteArr = bytearray(img_bytes)  # bytes2bytearray

    purchase = s
    s += ', %s' %(UUID)
    # {'topic': "example", 'payload':"example"} 發送圖片跟商品資料
    Topic2Edge = 'Edge/Purchase'
    Topic2Edge = os.path.join(Topic2Edge, UUID)
    print(Topic2Edge)
    msgs = [{'topic':"Database/Image", 'payload': byteArr},{'topic':"Database/Info", 'payload': s},{'topic':Topic2Edge, 'payload': purchase}]
    publish.multiple(msgs, hostname= "192.168.105.92", auth = {'username':"Test", 'password':"1234"})
        
    print('Done. (%.3fs)' % (time.time() - t0))

    # End Once Img------------------------------------------------


if __name__ == '__main__':
    #初始參數設定
    MQTT_SERVER = "192.168.105.92"  #140.122.105.184:8000
    MQTT_PATH = "Server/Image/#"
    # 連線設定
    client = mqtt.Client()
    # 設定登入帳號密碼
    client.username_pw_set("Test","1234")
    # 設定連線資訊(IP, Port, 連線時間)
    client.connect(MQTT_SERVER, 1883, 60)
    client.on_connect = on_connect
    client.on_message = on_message

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',nargs='+', type=str, default= ['runs/train/Shopv2/weights/best.pt'] , help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/samples/Images', help='source')  # file/folder, 0 for webcam ,Image or Mp4
    parser.add_argument('--output', type=str, default='Output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=448, help='inference size (pixels)') 
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='Object/cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/store.names', help='*.cfg path')

    opt = parser.parse_args()
    print(opt) #Setting Data Inf.

    # Get names and colors 取得名稱以及顏色
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # Initialize Switch GPU or CPU
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(opt.cfg, opt.img_size).cuda()

    try:
        model.load_state_dict(torch.load(opt.weights[0], map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        load_darknet_weights(model, opt.weights[0])

    model.to(device).eval()
    model.cuda()  # to FP16
    model.half()  # to FP16

    # MySQL
    MySQL_IP = 'localhost'
    MySQL_User, MySQL_Passwd = 'root', '1234'
    # MySQL Connect
    db = pymysql.connect(host= MySQL_IP,user= MySQL_User,password= MySQL_Passwd,database="ShopSystem")
    # 取得操作
    cursor = db.cursor()
    sql = "select * from ProductInfo"
    cursor.execute(sql)
    ProductName = np.array(cursor.fetchall())
    print(ProductName)
    db.commit()
    db.close()

    with torch.no_grad():
        client.loop_forever()

# COCO dataset: python3 MQTT_detect.py --weight models/weights/yolov4.weights --cfg models/cfg/yolov4.cfg --name data/coco.names
