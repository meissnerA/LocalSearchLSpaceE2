# Implementation of Keep it Simple: Evaluating Local Search-based Latent Space Editing
A reference will be added after a positive feedback from the reviewers

## Abstract: 
Semantic image editing allows users to selectively change entire image attributes in a controlled manner with just a few clicks. Most approaches use a generative adversarial network (GAN) for this task to learn an appropriate latent space representation and attribute-specific transformations. While earlier approaches often suffer from entangled attribute manipulations, newer ones improve on this aspect by using separate specialized networks for attribute extraction. Iterative optimization algorithms based on backpropagation constitute a possible approach to find attribute vectors with little entanglement. However, this requires a large amount of GPU memory, training instabilities can occur, and the used models have to be differentiable. To address these issues, we propose a local search-based approach for latent space editing. We show that it performs at the same level as previous algorithms and avoids these drawbacks.

## Notebook description:
- Following our paper reimplementation_enjoy_your_editing.ipynb contains a minimal reimplementation of Zhuang, P., Koyejo, O. O., and Schwing, A. (2021). Enjoy your editing: Controllable GANs for image editing via latent space navigation. In International Conference on Learning Representations https://arxiv.org/pdf/2102.01187.pdf. We have heavily borrowed from their implementation in https://github.com/KelestZ/Latent2im

- evaluate_latent_vecs.ipynb is also mostly the code from https://github.com/KelestZ/Latent2im/blob/main/eval.py, we have extened it by normalizing the length of attribute vectors such that the first element in interval_counter has the same amount of samples for each approach we compare

- our_approach_local_search.ipynb implements our approach to calculate attribute vectors for GANs

- For our comparison with shen we created 50,000 images (seed=0..49999), used the same regressor as in the notebooks to extract the attributes. Afterwards we used https://github.com/genforce/interfacegan to calculate the attribute vectors in StyleGAN2s w-Space.

## Requirements:
- Since we base our implementation on NVIDIAs stylegan2-ada-pytorch implementation just follow the instructions on https://github.com/NVlabs/stylegan2-ada-pytorch
- The easiest way is to install Anacoda with 64-bit Python 3.7, this provides jupyter notebook to open our Code
- Install Pytorch with pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
- Download the weights for StyleGAN2 [ffhq.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) and place the ffhq.pkl in the folder ./pretrained_models
