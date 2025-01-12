# Official LNT-YOLO

## Description

Implementation of paper - [LNT-YOLO: A Lightweight Nighttime Traffic Light Detection and Recognition Network](not yet published)

## Performance 

TN-TLD Dataset - can be downloaded via [Kaggle](https://kaggle.com/datasets/d025b632de3046a611bdda27bc3605ce8d0b7c3529764fd387fcfa78086d3684)

| Model | Test Size | mAP@0.5 | mAP@0.5:0.95 | GFLOPS | Params|
| :-- | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7-tiny**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt) | 1280 | **0.642** | **0.335** | 6 *M* | 13.1 |
| [**LNT-YOLO**](https://github.com/esmunir/LNT-YOLO/releases/download/weights/lnt_yolo.pt) | 1280 | **0.781** | **0.423** | 6.4 *M* | 14.9 |

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov7 -it -v your_coco_path/:/coco/ -v your_code_path/:/yolov7 --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# go to code folder
cd /yolov7
```

</details>

## Testing

[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt) [`lnt_yolo.pt`](https://github.com/esmunir/LNT-YOLO/releases/download/weights/lnt_yolo.pt)

``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```

## Training

Data preparation

* Download the TN-TLD data via [Kaggle](https://kaggle.com/datasets/d025b632de3046a611bdda27bc3605ce8d0b7c3529764fd387fcfa78086d3684) which contain the train, val, and test images and labels, and also its yaml files
* Extract the .zip files into /data folder

Single GPU training

``` shell
# train models
python train.py --workers 8 --device 0 --batch-size 32 --data data/tn-tld.yaml --img 1280 1280 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.lnt.yaml
```

Multiple GPU training

``` shell
# train models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/tn-tld.yaml --img 1280 1280 --cfg cfg/training/lnt-yolo.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.lnt.yaml
```

## Inference

On video:
``` shell
python detect.py --weights lnt_yolo.pt --conf 0.25 --img-size 1280 --source yourvideo.mp4
```

On image:
``` shell
python detect.py --weights lnt_yolo.pt --conf 0.25 --img-size 1280 --source inference/images/horses.jpg
```

## Export

**Pytorch to CoreML (and inference on MacOS/iOS)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7CoreML.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

**Pytorch to ONNX with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
```shell
python export.py --weights lnt_yolo.pt --grid --end2end --simplify \
--topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 1280 1280 --max-wh 1280
```

**Pytorch to TensorRT with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7trt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights ./lnt_yolo.pt--grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

**Pytorch to TensorRT another way** <a href="https://colab.research.google.com/gist/AlexeyAB/fcb47ae544cf284eb24d8ad8e880d45c/yolov7trtlinaom.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <details><summary> <b>Expand</b> </summary>


```shell
wget https://github.com/esmunir/LNT-YOLO/releases/download/weights/lnt_yolo.pt
python export.py --weights lnt_yolo.pt.pt --grid --include-nms
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o lnt_yolo.pt.onnx -e lnt_yolo.pt-nms.trt -p fp16

# Or use trtexec to convert ONNX to TensorRT engine
/usr/src/tensorrt/bin/trtexec --onnx=yolov7-tiny.onnx --saveEngine=lnt_yolo.pt-nms.trt --fp16
```

</details>

Tested with: Python 3.9.19, Pytorch 2.3.1+cu121

## Citation

```

```


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* This project is based on [YOLOv7](https://github.com/WongKinYiu/yolov7).

</details>
