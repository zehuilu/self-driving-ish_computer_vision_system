# How to save images from QLabs

Follow the documentation to start QCar in QLabs.

Open a terminal,
```
roslaunch qcar qcar.launch
```

To control the car manually, Open a terminal,
```
rosrun test_pkg commandnode_keyboard_old.py
```


- To save images for the RGBD camera, open a terminal,
```
cd <THIS_MAIN_DIRECTORY>
rosrun image_view image_saver image:=/qcar/rgbd_color _save_all_image:=false __name:=image_saver_rgb "_filename_format:=$PWD/images_qcar/RGBD/rgb_%04d.%s"
```

Open another terminal,
```
rosservice call /image_saver_rgb/save
```


- To save images from the front CSI camera, open a terminal,
```
cd <THIS_MAIN_DIRECTORY>
rosrun image_view image_saver image:=/qcar/csi_front _save_all_image:=false __name:=image_saver_front "_filename_format:=$PWD/images_qcar/FRONT_CSI/front_%04d.%s"
```

Open another terminal,
```
rosservice call /image_saver_front/save
```
