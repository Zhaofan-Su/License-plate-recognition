## 目标检测：车牌检测及字符串分析


概述
---

车牌识别在现实生活中的应用十分广泛，可以有效帮助交通管理部门获取车辆的相关信息，加强交通系统的监管。

车牌识别整体上可以分为两大部分：从图片中检测车牌以及对检测到的车牌进行字符串分割，最终获取车牌号码。借助于深度学习网络，我们可以高效地提高识别准确率，减少人工工作量。

本项目针对两个步骤构建了两个基本的CNN网络，在提供的数据集上进行训练检测。


项目运行方式
---

请使用下列命令运行项目：
```sh
git clont 
cd License-plate-recognition
python demo.py --char_model char-models/trained-model-name.meta --license_model license-models/license-model-name.meta
```
使用上述命令，你可以直接看到从给定的图片中分割出来的车牌区域以及对其进行字符串分割之后的识别效果。

字符识别CNN网络：

```sh
python cnn.py -t 0 -f 0     # 训练字符识别CNN网络
python cnn.py -t 0 -f 1 -m 'char-models/your model name'     # 测试字符识别CNN网络
```

车牌识别CNN网络：
```sh
pyton cnn.py -t 1 -f 0      # 训练车牌识别CNN网络
python cnn.py -t 1 -f 1 -m 'license-models/your model name'      # 测试车牌识别CNN网络
```

使用
```sh
python demo.py -h
python cnn.py -h 
```
查看命令行参数及取值