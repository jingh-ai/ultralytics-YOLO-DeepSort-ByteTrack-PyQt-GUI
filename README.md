# ultralytics-YOLO-DeepSort/ByteTrack-PyQt-GUI
a GUI application, which uses ultralytics YOLO for  Object Detection/Tracking, Human Pose Estimation/Tracking from images, videos or camera. 

All python scripts performing detection, pose and segmentation using the YOLO model in ONNX.

![GUI](./data/ui.png)

Supported AI tasks:
- [x] Detection
- [x] Pose Estimation
- [x] Segmentation
- [ ] OBB

Supported Models:
- [x] YOLO26
   - [x] YOLO26-n
   - [x] YOLO26-s
   - [x] YOLO26-m
   - [x] YOLO26-l
   - [x] YOLO26-x
- [x] YOLOv13
   - [x] YOLOv13-n
   - [x] YOLOv13-s
   - [x] YOLOv13-l
   - [x] YOLOv13-x
- [x] YOLO11
   - [x] YOLO11-n
   - [x] YOLO11-s
   - [x] YOLO11-m
   - [x] YOLO11-l
   - [x] YOLO11-x


Supported Trackers:
- [x] DeepSort
- [x] ByteTrack

Supported Input Sources:
- [x] local files: images or videos
- [x] Camera
- [x] RTSP-Stream

## Install

Install required packages with pip:

```shell
pip install -r requirements.txt
```

or with conda:

```shell
conda env create -f environment.yml

# activate the conda environment
conda activate yolo_gui
```

## Download weights

Download the model weights：

``````shell
python download_weights.py
``````

The model files are saved in the **weights/** folder.

## Run

```shell
python main.py
```

