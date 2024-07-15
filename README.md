# YOLOGANg_VehicleDetection

Our team's submission for the ICDEC'24 Vehicle Detection in Various Weather Conditions (VDVWC) Challenge.

## Dataset Format to Follow

The dataset is organized into training and validation sets with corresponding images and labels. The directory structure is as follows:

```
dataset/  
├── images/  
│   ├── train/  
│   └── val/  
└── labels/  
    ├── train/  
    └── val/
```




Please ensure that your dataset follows this structure for consistency and ease of use.

## Novelty of the method 

After training and testing many models from YOLOv8s, v8n, v10s, v10m, it was summarized that YOLOv10 performed the best on the given data. To further improve the performance metrics, we deployed Optuna to handle the finetuning of parameters. 
We initially tried training with the SDG optimizer, which when in comparison with the Adam optimizer gave poor results. After performing 10 trials, the best scores achieved were 0.543 mAP@0.5 IoU and 0.273 mAP@0.5:0.95 IoU. 
On this particular dataset, after experimenting it was found out that if augmentations were to be applied or synthetic data has to be created, the model turned out to overfit on the data and perform worse in general cases, so it was decided not to use augmentations apart from the 'randomaug' provided by YOLO and 'mosaic' and 'fliplr' being determined by Optuna.

We have introduced the addition of another block in the architecture, (not visible, because changes were made in code). To the output of every C2f block in the YOLO architecture, we added a Squeeze and Excitation block (SE Block) which improves attention mechanism of the model according to some papers in 2019. This was done in the hopes that the model distinguishes better between the background and named classes. Although, the effect is minimal (0.267 mAP@0.5:0.95 to 0.273 mAP@0.5:0.95), it is still useful.

In just around 5 to 10 epochs, once the correct hyperparameters were found by Optuna, we were able to take the pretrained yolov10m.pt model and make it achieve the mentioned metrics. Training time was only around 10-12mins, once hyperparameters were foudn on a free-tier Colab T4 GPU, and the model is lightweight, with it's 'pt' file being only 31.9mb in size.

Optuna was used with persistent memory storage so that even later, the database used by optuna to find the right parameters can be accessed. This makes it future proof for better fine-tuning.

## Observations 

The strategy of using single-vehicle images of underrepresented classes and overlaying them on some road backgrounds in various weather conditions, ironically reduced the usability of the model after training, because there wasn't enough variety of images/etc. To do this, we extracted images of underrepresented classes from the dataset itself by cropping them out of their backgrounds, processing it with rembg to remove background and overlaying it on images. THis worsened the results as bounding boxes were being drawn in irrelevant places.

Another strategy used was to deploy CycleGAN to achieve style-transfer, (day to night) and (night to day) to virtually expand the data given to us. But the results of this use were not good, as the model was overfitting. If there was more variation in data, then the above two methods could have helped. 

## Changes made to architecure (in code) 

1. Created a new file se_block.py in the ultralytics/nn/modules/ directory, implementing the SEBlock class.

2. Modified the C2f class in ultralytics/nn/modules/block.py to incorporate the SEBlock.

Imported the SEBlock class
Added an SE block after the convolution operations in the forward pass
The SE block was integrated into the C2f module, which is a core component of the YOLOv10 architecture, affecting all variants.

3. No changes were made to the YAML configuration files, as the modification was done at the module level.

4. The modified Ultralytics package was reinstalled using pip install -e . in the package directory.

This change affects all YOLO models when using the modified package, not just YOLOv10m. 

Modified C2f block in ultralytics/ultralytics/nn/modules/block.py. 
```
class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.se = SEBlock(c2)  # Add SE block here

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.se(self.cv2(torch.cat(y, 1)))  # Apply SE block to the output

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.se(self.cv2(torch.cat(y, 1)))  # Apply SE block to the output
```

se_block.py at ultralytics/ultralytics/nn/modules/se_block.py. 
```
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```
