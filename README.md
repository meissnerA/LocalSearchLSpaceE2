# Implementation of Keep it Simple: Evaluating Local Search-based Latent Space Editing
A reference will be added after a positive feedback from the reviewers

## Notebook description:
- The notebook main.ipynb describes which script produced which result of the paper. We added main.html as an alternative if the notebook is not displayed in your browser.
- For our comparison with shen we created 50,000 images (seed=0..49999), used the same regressor as in the notebooks to extract the attributes. Afterwards we used https://github.com/genforce/interfacegan to calculate the attribute vectors in StyleGAN2s w-Space.
- the files in stylegan2-stylespace are a subset of https://github.com/orpatashnik/StyleCLIP/tree/main/global_torch

## Requirements:
- Since we base our implementation on NVIDIAs stylegan2-ada-pytorch implementation just follow the instructions on https://github.com/NVlabs/stylegan2-ada-pytorch
- The easiest way is to install Anacoda with 64-bit Python 3.7, this provides jupyter notebook to open our Code
- Install Pytorch with pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
- Download the weights for StyleGAN2 [ffhq.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) and place the ffhq.pkl in the folder ./pretrained_models
