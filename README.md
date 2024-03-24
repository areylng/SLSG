# SLSG: Industrial Image Anomaly Detection by Learning Better Feature Embeddings and One-Class Classification
## Introduction
Industrial image anomaly detection under the setting of one-class classification has significant practical value. However, most existing models face challenges in extracting separable feature representations when performing feature embedding and in constructing compact descriptions of normal features when performing one-class classification. One direct consequence of this is that most models perform poorly in detecting logical anomalies which violate contextual relationships. Focusing on more effective and comprehensive anomaly detection, we propose a network based on self-supervised learning and self-attentive graph convolution (SLSG). SLSG uses a generative pre-training network to assist the encoder in learning the embedding of normal patterns and the reasoning of position relationships. Subsequently, SLSG introduces the pseudo-prior knowledge of anomaly through simulated abnormal samples. By comparing the simulated anomalies, SLSG can better summarize the normal patterns and narrow down the hypersphere used for one-class classification. In addition, with the construction of a more general graph structure, SLSG comprehensively models the dense and sparse relationships among elements in the image, which further strengthens the detection of logical anomalies. Extensive experiments on benchmark datasets show that SLSG achieves superior anomaly detection performance, demonstrating the effectiveness of our method.
![](C:\Users\Lenovo\Desktop\异常检测——论文2\高清截图_NEW\2023-04-14_171559.png)

## Datasets
Note the change of  dataset path in the program.
- [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco)
- [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

To calculate the image-level ROAUC scores for the MVTec LOCO AD dataset, the LOCO AD dataset needs to be organized into the following format:
```
MVTec LOCO AD dataset
|—— breakfast_box
|   |—— ground_truth
|	|	|-- logical_anomalies
|   |   	|—— 000_mask.png
|   |   	|—— 001_mask.png
|	|	|-- structural_anomalies
|   |   	|—— 000_mask.png
|   |   	|—— 001_mask.png
|   |—— test
|	|	|-- good
|   |   	|—— 000.png
|   |   	|—— 001.png
|	|	|-- logical_anomalies
|   |   	|—— 000.png
|   |   	|—— 001.png
|	|	|-- logical_anomalies
|   |   	|—— 000.png
|   |   	|—— 001.png
|   |—— train
|	|	|-- good
|   |   	|—— 000.png
|   |   	|—— 001.png
|—— juice_bottle
|   |—— ...
|   |—— ...
```
## Training
```
python run_training.py
```
## Inference
```
python run_inference.py
```

We provide weights for model inference：https://drive.google.com/drive/folders/18VTqznST_fbebgtyu_w8nfbYvVFiNaYj?usp=drive_link
Image-level evaluation metrics (AUROC) of the provide model weights on the MVTec LOCO AD dataset:

|            | Breakfast box | Juice bottle | Pushpins | Screw bag | Splicing connector | Mean  |
| :--------: | :-----------: | :----------: | :------: | :-------: | :----------------: | :---: |
| Structural |     0.829     |    0.977     |  0.941   |   0.932   |       0.935        | 0.923 |
|  Logical   |     0.944     |    0.991     |  0.967   |   0.734   |       0.862        | 0.899 |
|    Mean    |     0.884     |    0.985     |  0.955   |   0.808   |       0.894        | 0.905 |

## Reference Codes
- https://github.com/LilitYolyan/CutPaste
- https://github.com/VitjanZ/DRAEM
- https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master
