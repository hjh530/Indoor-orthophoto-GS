# 本项目是针对室内3DGS生成正射影像的教程和重要的一些代码

## 由colmap开源软件生成稀疏点云和相机位姿

数据输入格式参照3DGS格式，导出为.txt格式
<div align="center">
  <img src="./assets/colmap.png" />
</div>

## 坐标系转化
代码位于[坐标系转化生成虚拟视角](./create_virtual_camera.py)，主要是利用ransack算法估计地面平面，将世界坐标系的z轴转化到地面平面上，方向朝上，然后通过XY 轴曼哈顿对齐 (Manhattan World Alignment)，房间一般都是矩形，已知z轴即可通过降维将点全部投影到xoy平面中然后通过最小包围盒算法找到x，y轴方向，xy轴方向需要与房间的墙面平行，所以该算法适合于矩形房间。
<div align="center">
  <img src="./assets/colmap2.png" />
</div>

## 准备3DGS训练数据
代码位于[生成占位图像](./create_dummy_images.py)，为了顺利进入3DGS训练，需要生成虚拟视角的占位图，这些图像不需要参与训练，只需要渲染即可。

## 训练3DGS
这部分与[3DGS](https://github.com/graphdeco-inria/gaussian-splatting)训练逻辑一致，但是加入的虚拟视角是不需要参与训练的，所以对[3DGS训练代码](./train.py)进行修改，这个demo主要是根据另外一个项目写的，这个项目同样适用，可以忽略添加的其他一些参数。
