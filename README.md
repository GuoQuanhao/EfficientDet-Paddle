## 数据文件准备

数据集已挂载至aistudio项目中，如果需要本地训练可以从这里下载[数据集](https://aistudio.baidu.com/aistudio/datasetdetail/103218)，项目已同步至[AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/2444051)

工程目录大致如下，可根据实际情况修改
```
home/aistudio
|-- coco(dataset)
|   |-- annotions
|       |-- instances_train2017.json
|       |-- instances_val2017.json
|   |-- train2017
|   |-- val2017
|   |-- test2017
|-- EfficientDet(repo)

```

## 训练

### 单卡训练
```
python train.py -c 0 -p coco --batch_size 8 --lr 1e-5
```
`-c X`表明`efficientdet-dX`
![](https://ai-studio-static-online.cdn.bcebos.com/f41ba6a639be4afca746c731178b61a9e03719f9fb2f46c49544038e185c1acd)



### 验证

确保已安装`pycocotools`和`webcolors`
```
pip install pycocotools webcolors
```
```
python coco_eval.py -p coco -c 0
```
你需要将权重文件下载至`weights`文件夹下，或者使用`-w`手动指定权重路径

#### 验证结果如下所示
**所有完整验证结果可在`EfficientDet/benchmark/coco_eval_result.txt`下查看**

| coefficient | pth_download | GPU Mem(MB) |mAP 0.5:0.95(this repo) | mAP 0.5:0.95(official) |
| :-----: | :-----: | :------: | :------: | :------: |
| D0 | [efficientdet-d0.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d0.pdparams) | 1049 |33.1 | 33.8
| D1 | [efficientdet-d1.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d1.pdparams) | 1159 |38.8 | 39.6
| D2 | [efficientdet-d2.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d2.pdparams) | 1321 |42.1 | 43.0
| D3 | [efficientdet-d3.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d3.pdparams) | 1647 |45.6 | 45.8
| D4 | [efficientdet-d4.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d4.pdparams) | 1903 |48.5 | 49.4
| D5 | [efficientdet-d5.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d5.pdparams) | 2255 |50.0 | 50.7
| D6 | [efficientdet-d6.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d6.pdparams) | 2985 |50.7 | 51.7
| D7 | [efficientdet-d7.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d7.pdparams) | 3819 |52.6 | 53.7
| D7X | [efficientdet-d8.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d8.pdparams) | 3819 |53.8 | 55.1

### 推理

```
python efficientdet_test.py
```
注意到你需要手动更改中第`17`行`compound_coef = 8`来指定`efficientdet-dX`

**部分模型推理结果如下所示**

<div align="center">
  <font face="楷体" size=4>原始图像&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;官方</font><font face="Times New Roman" size=4>efficientdet-d0</font><font face="楷体" size=4>预测图像</font>
    
  <img src="https://ai-studio-static-online.cdn.bcebos.com/086552e84f5647888373676612f34b583e27a8a63386457f971e5cdb824d964b" width="450"/><img src="https://ai-studio-static-online.cdn.bcebos.com/701aca4bbfc8410b8c7d5d824ae93dbb885eaacb115347458f579f201ee18088" width="450"/>
</div>


<div align="center">
  <font face="楷体" size=4>本项目</font><font face="Times New Roman" size=4>efficientdet-d0</font><font face="Times New Roman" size=4>预测图像&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本项目</font><font face="Times New Roman" size=4>efficientdet-d8</font><font face="Times New Roman" size=4>预测图像</font>
    
  <img src="https://ai-studio-static-online.cdn.bcebos.com/c063b3deaa1a42b1abee5dcb52c1ab8e1c74e557802341ce92c7f6528e098de4" width="450"/><img src="https://ai-studio-static-online.cdn.bcebos.com/c2d3a5a474cd4da9942af092c05c38f9af0a12551e8b4812981852d54154c460" width="450"/>
</div>

```
python efficientdet_test_videos.py
```

**以`efficientdet-d0`为例，测试效果如下**

![results](https://user-images.githubusercontent.com/49911294/136463881-928ee08f-6a03-4966-9b22-7e224523c813.gif)


### TODO

- **多卡训练(Coming soon)**
- **EfficientNet Pretrained Weights(Coming soon)**

#### [GitHub地址](https://github.com/GuoQuanhao/EfficientDet-Paddle)

# **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| BLOG主页        | [DeepHao的 blog 主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
| GitHub主页        | [DeepHao的 github 主页](https://github.com/GuoQuanhao) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
