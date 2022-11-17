# CAMUS Segmentation: Semantic segmentation with PyTorch using an open large-scale dataset in 2D Echocardiography 

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-v3.6+-9cf.svg?logo=python&style=flat" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.10.1-red.svg?logo=PyTorch&style=flat" /></a>
<a href="https://releases.ubuntu.com/18.04.6/?_ga=2.264946639.1101608567.1668677170-1638439465.1665151863"><img src="https://img.shields.io/badge/Ubuntu-18.04-informational.svg?logo=Ubuntu&style=flat" /></a>
<a href="https://github.com/GKalliatakis/camus-segmentation-pytorch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-orange.svg" /></a>


## Detailed architectures
<p align="justify">
The two U-Net implementations presented in the paper are summarised in tables I and II below. 
Both differ from the original U-Net proposed by Ronneberger et al. 
U-Net 1 is optimised for speed, while U-Net 2 is optimized for accuracy. 
U-Net 1 shows a more compact architecture, with the addition of one downsampling level. 
In the U-Net 2 design, the number of filters per convolutional layers
increases and decreases linearly from an initial kernel size of 48, which makes for a wider net.
</p>

<p align="center">
  <img width="892" src="figures/detailed_architectures.png" alt="detailed_architectures">
</p>

<p align="center">
  <img width="892" src="figures/U-Net2.png" alt="U-Net2">
</p>

## Usage
    
```python
import torch
from camus_unet.camus_unet1 import CamusUnet1

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CamusUnet1()  # initialise the U-NET 1 model

# move initialised model to chosen device
model = model.to(device)

# the usual training loop goes here...
```

## References
- [Deep Learning for Segmentation using an Open  Large-Scale Dataset in 2D Echocardiography](https://arxiv.org/pdf/1908.06948.pdf)
- [CAMUS project](https://www.creatis.insa-lyon.fr/Challenge/camus/)
- [CAMUS Exploratory Data Analysis](https://www.kaggle.com/code/sontungtran/camus-eda/notebook)
- [PyTorch U-Net](https://github.com/milesial/Pytorch-UNet)
- [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)

## License
Distributed under the MIT License. See [LICENSE](https://github.com/GKalliatakis/camus-segmentation-pytorch/blob/main/LICENSE) file for more information.
