https://user-images.githubusercontent.com/11009876/132947317-3c839522-b347-4a8d-8675-3999adf9cdb6.mp4

# Self-Driving-ish Computer Vision System
- This project generates images you've probably seen in autonomous driving demo
- Detection
    - Object Detection and Tracking
    - Lane Detection and Curve Fitting
    - Road Segmentation
    - Depth Estimation
- Transform using Projective Geometry and Pinhole Camera Model
    - Normal View -> Top View
    - Distance Calculation (image plane -> ground plane in world coordinate system)

## Result
- YoutTube: https://youtu.be/GB4p_fjQZNE

![result](00_doc/result.jpg)


# Tested Environment
## Computer
- Windows 10 (x64) + Visual Studio 2019
    - Intel Core i7-6700 @ 3.4GHz + NVIDIA GeForce GTX 1070
- Jetson Xavier NX. JetPack 4.6
    - You will get error if using JetPack 4.5 or before because of TensorRT error

## Deep Learning Inference Framework
- TensorFlow Lite with XNNPACK delegate
    - CPU
    - Note: Running with CPU is very slow
- TensorRT
    - GPU

# Usage
```
./main [input]
 - input:
    - use the default image file set in source code (main.cpp): blank
        - ./main
     - use video file: *.mp4, *.avi, *.webm
        - ./main test.mp4
     - use image file: *.jpg, *.png, *.bmp
        - ./main test.jpg
    - use camera: number (e.g. 0, 1, 2, ...)
        - ./main 0
    - use camera via gstreamer on Jetson: jetson
        - ./main jetson
```

- Mouse Drag: Change top view angle
- Keyboard (asdwzx) : Change top view position


# How to build a project
## 0. Requirements
- OpenCV 4.x
- CMake
- TensorRT 8.0.x
    - If you get build error related to TensorRT, modify cmake settings for it in `inference_helper/inference_helper/CMakeLists.txt`

## 1. Download source code and pre-built libraries 
- Download source code
    - If you use Windows, you can use Git Bash
    ```sh
    git clone https://github.com/iwatake2222/self-driving-ish_computer_vision_system.git
    cd self-driving-ish_computer_vision_system
    git submodule update --init
    sh inference_helper/third_party/download_prebuilt_libraries.sh
    ```
- Download models
    ```sh
    sh ./download_resource.sh
    ```

## 2-a. Build in Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2019 64-bit
    - `Where is the source code` : path-to-cloned-folder
    - `Where to build the binaries` : path-to-build	(any)
- Open `main.sln`
- Set `main` project as a startup project, then build and run!
- Note:
    - You may need to modify cmake setting for TensorRT for your environment

## 2-b. Build in Linux (Jetson Xavier NX)

Added by me:

First, make sure that you install NVIDIA Graphics Drivers.
You can check by
```
nvidia-smi
```

1. Install OpenCV on Linux
```
sudo apt install libopencv-dev
```

2. Install CUDA on Linux

- Install by package manager (Only cuda 10)
```
sudo apt update
sudo apt install build-essential
# sudo apt install nvidia-cuda-toolkit

sudo apt-get install linux-headers-$(uname -r)
```

- Install by following NVIDIA's official instructions (cuda 11 or 12)

Download the NVIDIA CUDA Toolkit from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
Select options work for you.
Then you will see some commands similar to the following. Run these commands.

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

Reference: [https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

3. Install cuDNN on Linux
```
sudo apt install zlib1g
```

```
wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
```
, where `${OS}` is `debian11`, `ubuntu1804`, `ubuntu2004`, or `ubuntu2204`.

My case is
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
```

```
sudo apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
sudo apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
sudo apt-get install libcudnn8-samples=${cudnn_version}-1+${cuda_version}
```
, where `${cudnn_version}` is `8.9.1.*`, `${cuda_version}` is `cuda12.1` or `cuda11.8`.




Reference:
[https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux)


4. How to install TensorRT:
```

```


Build this repo:
```sh
mkdir build && cd build
# cmake .. -DENABLE_TENSORRT=off
cmake .. -DENABLE_TENSORRT=on
make
./main
```

# Note
## cmake options
```sh
cmake .. -DENABLE_TENSORRT=off  # Use TensorFlow Lite (default)
cmake .. -DENABLE_TENSORRT=on   # Use TensorRT

cmake .. -DENABLE_SEGMENTATION=on    # Enable Road Segmentation function (default)
cmake .. -DENABLE_SEGMENTATION=off   # Disable Road Segmentation function

cmake .. -DENABLE_DEPTH=on    # Enable Depth Estimation function (default)
cmake .. -DENABLE_DEPTH=off   # Disable Depth Estimation function
```

## Misc
- It will take very long time when you execute the app for the first time, due to model conversion
    - I took 80 minutes with RTX 3060ti
    - I took 10 - 20 minutes with GTX 1070

# Software Design
## Class Diagram
![class_diagram](00_doc/class_diagram.jpg)

## Data Flow Diagram
![data_flow_diagram](00_doc/data_flow_diagram.jpg)

# Model Information
## Details
- Object Detection
    - YOLOX-Nano, 480x640
    - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano_new.sh
    - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano.sh
- Lane Detection
    - Ultra-Fast-Lane-Detection, 288x800
    - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/140_Ultra-Fast-Lane-Detection/download_culane.sh
- Road Segmentation
    - road-segmentation-adas-0001, 512x896
    - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/136_road-segmentation-adas-0001/download.sh
- Depth Estimation
    - LapDepth, 192x320
    - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/148_LapDepth/download_ldrn_kitti_resnext101.sh
    - LapDepth, 256x512
    - [00_doc/pytorch_pkl_2_onnx_LapDepth.ipynb](00_doc/pytorch_pkl_2_onnx_LapDepth.ipynb)

## Performance
| Model                            | Jetson Xavier NX | GTX 1070 |
| -------------------------------- | ---------------: | -------: |
| == Inference time ==                                           |
|  Object Detection                |          10.6 ms |   6.4 ms |
|  Lane Detection                  |           9.6 ms |   4.9 ms |
|  Road Segmentation               |          29.1 ms |  13.5 ms |
|  Depth Estimation                |          55.2 ms |  37.8 ms |
| == FPS ==                                                      |
|  Total (All functions)           |          6.8 fps | 10.9 fps |
|  Total (w/o Segmentation, Depth) |         24.4 fps | 33.3 fps |

* Input
    - Jetson Xavier NX: Camera
    - GTX 1070: mp4 video
* With TensorRT FP16
* "Total" includes image read, pre/post process, other image process, result image drawing, etc.

# License
- Copyright 2021 iwatake2222
- Licensed under the Apache License, Version 2.0
    - [LICENSE](LICENSE)


# Acknowledgements
I utilized the following OSS in this project. I appreciate your great works, thank you very much.

## Code, Library
- TensorFlow
    - https://github.com/tensorflow/tensorflow
    - Copyright 2019 The TensorFlow Authors
    - Licensed under the Apache License, Version 2.0
    - Generated pre-built library
- TensorRT
    - https://github.com/nvidia/TensorRT
    - Copyright 2020 NVIDIA Corporation
    - Licensed under the Apache License, Version 2.0
    - Copied source code
- cvui
    - https://github.com/Dovyski/cvui
    - Copyright (c) 2016 Fernando Bevilacqua
    - Licensed under the MIT License (MIT)
    - Copied source code

## Model
- PINTO_model_zoo
    - https://github.com/PINTO0309/PINTO_model_zoo
    - Copyright (c) 2019 Katsuya Hyodo
    - Licensed under the MIT License (MIT)
    - Copied converted model files
- YOLOX
    - https://github.com/Megvii-BaseDetection/YOLOX
    - Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved
    - Licensed under the Apache License, Version 2.0
- Ultra-Fast-Lane-Detection
    - https://github.com/cfzd/Ultra-Fast-Lane-Detection
    - Copyright (c) 2020 cfzd
    - Licensed under the MIT License (MIT)
- road-segmentation-adas-0001
    - https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/road-segmentation-adas-0001
    - Copyright (c) 2021 Intel Corporation
    - Licensed under the Apache License, Version 2.0
- LapDepth-release
    - https://github.com/tjqansthd/LapDepth-release
    - Licensed under the GNU General Public License v3.0

## Image Files
- OpenCV
    - https://github.com/opencv/opencv
    - Licensed under the Apache License, Version 2.0
- YoutTube
    - https://www.youtube.com/watch?v=tTuUjnISt9s
    - Licensed under the Creative Commons license
    - Copyright Dashcam Roadshow 2020
