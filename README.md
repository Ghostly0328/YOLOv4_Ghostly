# YOLOv4

This is PyTorch implementation of [YOLOv4](https://github.com/AlexeyAB/darknet) which is based on [[YOLOv4_WongKinYiu on github]](https://github.com/WongKinYiu/PyTorch_YOLOv4).

* [[ultralytics/yolov3]](https://github.com/ultralytics/yolov3)

* [[original Darknet implementation of YOLOv4]](https://github.com/AlexeyAB/darknet)

* [[ultralytics/yolov5 based PyTorch implementation of YOLOv4]](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u5).

### 開發日誌

<details><summary> <b>展開</b> </summary>

* `2022-06-28` - 分支並上傳
* `2023-08-09` - 上傳最後版本
* `2023-08-10` - 修改README.MD 

</details>

## 安裝需要軟體 Requirements


```bash
pip install -r requirements.txt
```

下載權重檔(.weight)和模型(.cfg)並放到 ./models/weights/ 和 ./models/cfg/

※ For running Mish models, please install https://github.com/thomasbrandon/mish-cuda

## 檢測 Detect

```bash
python detect.py -h #使用說明
python detect.py --cfg ./models/cfg/yolov3.cfg --weights ./models/weights/yolov3_320.weights --source ./data/samples/ 
```

## 訓練 Training

```bash
python train.py --data coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights '' --name yolov4-pacsp
```

## 測試 Testing

```bash
python test.py --img 640 --conf 0.001 --data coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights weights/yolov4-pacsp.pt
```

## 引用 Citation

```
@article{bochkovskiy2020yolov4,
  title={{YOLOv4}: Optimal Speed and Accuracy of Object Detection},
  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2004.10934},
  year={2020}
}
```

```
@inproceedings{wang2020cspnet,
  title={{CSPNet}: A New Backbone That Can Enhance Learning Capability of {CNN}},
  author={Wang, Chien-Yao and Mark Liao, Hong-Yuan and Wu, Yueh-Hua and Chen, Ping-Yang and Hsieh, Jun-Wei and Yeh, I-Hau},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={390--391},
  year={2020}
}
```

## Acknowledgements

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
