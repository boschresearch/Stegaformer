# Effective Message Hiding with Order-Preserving Mechanisms

Code is under internal approval process. Coming soon.

Official implementation of "Effective Message Hiding with Order-Preserving Mechanisms" (BMVC 2024) [[Arxiv](https://arxiv.org/abs/2402.19160)].

Our paper introduces a new message-hiding model named **Stegaformer**. We designed this model to achieve very high recovery accuracy (greater than 99.9%) while also maintaining leading image quality. The code allows users to reproduce and extend the results reported in the study. Please cite the paper above when reporting, reproducing, or extending the results.

<center>
  <img src=https://github.com/boschresearch/Stegaformer/blob/main/figure/main.png width=60% />
</center>

## Purpose of the Project

This software is a research prototype, developed solely for and published as part of the publication cited above. It will not be maintained or monitored in any way.

## Installation and Requirements
### Setup

git clone https://github.com/boschresearch/Stegaformer.git

TThe main modules are built upon timm 0.6.2.dev0. The code is tested under PyTorch 2.0.0 with CUDA 11.8. To ensure fairness in the comparison of image quality, we use torchmetrics 0.11.4 for the measurement of PSNR and SSIM.

## Training and Validation
### Datasets
Our model is tested using randomly selected images from the COCO dataset and the full set of Div2k. You can find .txt files in /data/coco containing all the file names of our sampled COCO images.

### Training
We provide the training logs for the COCO dataset for our 1, 2, 3, 4, 6, and 8 BPP models in ./stega_model. Most of the logs show slightly better performance compared to the numbers listed in the paper, as experiment time was limited when meeting the deadline. You can use the following commands to train your own models:

**1BPP**

python train.py --batch_size 4 --bpp 1 --query im --msg_scale 2 --msg_range 1 --msg_pose 2d --data_path /media/SSD2/stega_all/stega_data/ --dataset coco --base_lr 0.0005 --iterations 80000

**2BPP**

python train.py --batch_size 4 --bpp 2 --query im --msg_scale 2 --msg_range 1 --msg_pose 2d --data_path /media/SSD2/stega_all/stega_data/ --dataset coco --base_lr 0.00025 --iterations 80000

**3BPP**

python train.py --batch_size 4 --bpp 3 --query im --msg_scale 2 --msg_range 1 --msg_pose 2d --data_path /media/SSD2/stega_all/stega_data/ --dataset coco --base_lr 0.000125 --iterations 80000

**4BPP**

python train.py --batch_size 4 --bpp 4 --query im --msg_scale 2 --msg_range 1 --msg_pose 2d --data_path /media/SSD2/stega_all/stega_data/ --dataset coco --base_lr 0.0001 --iterations 80000

**6BPP**

python train.py --batch_size 4 --bpp 3 --query im --msg_scale 2 --msg_range 3 --msg_pose 2d --data_path /media/SSD2/stega_all/stega_data/ --dataset coco --base_lr 0.00005--iterations 160000

**8BPP**

python train.py --batch_size 4 --bpp 2 --query im --msg_scale 2 --msg_range 15 --msg_pose 2d --data_path /media/SSD2/stega_all/stega_data/ --dataset coco --base_lr 0.00005 --iterations 160000

As described in the paper, the --bpp flag controls the length of the message segment, and --msg_range sets the range of the message elements (usually set to 1). You can select the types of positional embeddings using --msg_pose. There are three options: '2d', '1d', and 'learn'. Both '2D' and '1D' positional embeddings use fixed positional embedding, while the 'learn' type uses a learned positional embedding.

## License

**Stegaformer** is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open-source components included in **Stegaformer**, see the 3rd-party-licenses.txt file.

## Purpose of the project
This software is a research prototype, solely developed for and published as part of the publication cited above.

## Contact 
Please feel free to open an issue or contact us personally if you have questions, need help, or need explanations. Don't hesitate to write an email to the following address: yu.gao2@cn.bosch.com.

