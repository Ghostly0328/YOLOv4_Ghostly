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
* `2023-08-12` - 簡化資料集路徑，添加docker操作教學

</details>

## 安裝要求 Installation Requirements

Ubuntu:  
確保先安裝 Anaconda，然後執行以下命令

```bash
git clone https://github.com/Ghostly0328/YOLOv4_Ghostly.git
conda env create -f environments.yml
conda activate yolov4
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
docker (建議):  
記得需要先安裝具有GPU的docker，可以參考這個[網址](https://hackmd.io/@joshhu/Sy8MQetvS)來測試
```bash
docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
docker run --name yolov4 -it -v /git_path/:/yolo --workdir /yolo --rm --gpus 0 --shm-size=64g pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
apt update
apt install -y zip htop screen libgl1-mesa-glx
apt-get install gcc libglib2.0-0
pip install -r requirements.txt
```

下載權重檔(.weight)和數據集並放到 ./models/weights/ 和 ./data/dataset/

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
