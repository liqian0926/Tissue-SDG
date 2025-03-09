# Tissue-SDG (the Visual Computer)
This repository provides the official PyTorch implementation of the following paper:
> [**Dynamic Adaptive Data Augmentation and Multi-Scale Contrastive Learning for Generalized Tissue Semantic Segmentation**]<br>
> Qian Li, Jiayi Peng, Jiatai Lin, Chu Han, Ke Zhao, Zaiyi Liu<br>


> **Abstract:**
> *Tissue semantic segmentation of pathological images is critical in computational pathology. Deep learning models have shown promise in this task but often suffer from performance degradation when encountering out-of-distribution data due to variations in multi-center data distributions. To address this issue, we propose a dynamic single-domain generalization framework named Tissue-SDG for tissue semantic segmentation. Tissue-SDG incorporates dynamic adaptive data augmentation (DyADA) and multi-scale contrastive learning (MSCL) within a supervised framework to guide the model in learning domain-invariant features. Extensive experiments on our in-house CRC-GDPH-MS-TissueSeg dataset demonstrate the effectiveness of Tissue-SDG, which outperforms existing methods. Specifically, Tissue-SDG achieves an average Mean IoU of 79.64% across different test domains, exhibiting a remarkable improvement of at least 1% over baseline approaches. Here we show that our proposed techniques enhance the model's generalization ability and stability, making it more suitable for real-world applications.* <br>

### Installation
Clone this repository.
```
git clone https://github.com/liqian0926/Tissue-SDG.git
cd Tissue-SDG
```
Install dependencies by:
```
pip install -r requirements.txt
```

### How to Run Tissue-SDG
We evaluated the model on our in-house multi-scanner dataset CRC-GDPH-MS-TissueSeg. The structure of the data is as follows：

```
CRC-GDPH-MS-TissueSeg
 └ train
   └ train
     └ img
     └ mask
     └ train.txt
   └ val
     └ img
     └ mask
     └ val.txt
 └ test
    └ img
    └ mask
    └ text.txt
```
Modify the args.dataroot and args.dataset in the code to use your own dataset for training.  
Run the following command to train Tissue-SDG:
```bash
python train.py 
```
Run the following command to evaluate Tissue-SDG:
```bash
python pred.py 
```


