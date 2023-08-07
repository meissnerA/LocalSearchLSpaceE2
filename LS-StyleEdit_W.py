stylegan2_path = "./stylegan2-ada-pytorch"

import sys
sys.path.append(stylegan2_path)

import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import os
import click
import dnnlib
import legacy
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="optimizing attribute vector for StyleGAN2")
parser.add_argument("device", metavar="device", type=str, help="enter torch device")
parser.add_argument("sampleRadius", metavar="sampleRadius", type=float, help="scales the amount of noise used in the local search")
parser.add_argument("target_feature_index", metavar="target_feature_index", type=int, help="index of a target attribute according to Celeba, 31=Smiling, 9=Hair color")
parser.add_argument("maxLength", metavar="maxLength", type=str, help="the maximum Length a attribute vector before we normalize, usually values < 20 work fine")
parser.add_argument("batch_size", metavar="batch_size", type=str, help="how many samples to average over, 1-8 worked all fine in our experiments")
parser.add_argument("output_path", metavar="output_path", type=str, help="path to save the attribute vectors to e.g. /training_runs/stylegan2")


parser_args = parser.parse_args()
device = parser_args.device

size = 1024
truncation_psi = 0.5
noise_mode = 'const'
network_pkl = './pretrained_models/ffhq.pkl'
n_iters = 20001

with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

label = torch.zeros([1, G.c_dim], device=device)


z = torch.from_numpy(np.random.RandomState(0).randn(1, G.z_dim)).to(device)
w0 = G.mapping(z,label, truncation_psi=truncation_psi)

for random_seed in range(1, 1000):
    z = torch.from_numpy(np.random.RandomState(random_seed).randn(1, G.z_dim)).to(device)
    w = G.mapping(z,label, truncation_psi=truncation_psi)
    w0 = torch.cat([w0, w], axis=0)

feature_normalization = w0.std(axis=0).to(device)

file_to_read = open("./pretrained_models/resnet_092_all_attr_5_epochs.pkl", "rb")
regressor = pickle.load(file_to_read)
file_to_read.close()
regressor = regressor.to(device)
regressor.eval()

time1 = time.time()

def run_training(device, batch_size, maxLength, n_iters, target_feature_index, sampleRadius):
    d = torch.zeros([1, 512]).to(device)
    logging_folder = stylegan2_path + output_path + f'/our_approach_feature_{target_feature_index}_maxLenght_{maxLength}_lr_{sampleRadius}_batch_size{batch_size}'
    if not os.path.exists(logging_folder):
        os.makedirs(logging_folder)

    if not os.path.exists(logging_folder + '/saved_latent_vecs'):
        os.makedirs(logging_folder + '/saved_latent_vecs')

    with open(logging_folder + "/training_log.txt", "a") as myfile:
        myfile.write(f'Hyperparameters: feature_{target_feature_index} maxLenght_{maxLength} sampleRadius_{sampleRadius} batch_size{batch_size}' + '\n')

    for random_seed in range(n_iters):
        with torch.no_grad():
            z = torch.from_numpy(np.random.RandomState(random_seed).randn(batch_size, 512)).to(device)
            w = G.mapping(z, label, truncation_psi=truncation_psi)

            epsilon = np.random.choice([-1,1])
            img_d = G.synthesis(w+(d*epsilon), noise_mode=noise_mode)
            img_d = F.interpolate(img_d, size=256)
            alpha_d = regressor(img_d)[:, target_feature_index]

            d_new = d + sampleRadius * feature_normalization * torch.Tensor(np.random.RandomState(random_seed).randn(1, 512)).to(device)
            if torch.norm(d_new/feature_normalization).item() > maxLenght:
                d_new = maxLenght*d_new/torch.norm(d_new/feature_normalization)

            img_d_new = G.synthesis(w+(d_new*epsilon), noise_mode=noise_mode)
            img_d_new = F.interpolate(img_d_new, size=256)
            alpha_d_new = regressor(img_d_new)[:, target_feature_index]

            print('pred_old:', alpha_d.mean().item(), 'pred_new:', alpha_d_new.mean().item())
            with open(logging_folder + "/training_log.txt", "a") as myfile:
                myfile.write('pred_old: ' + str(alpha_d.mean().item()) + 'pred_new: ' + str(alpha_d_new.mean().item()) + '\n')

            if random_seed%1000 == 0:
                torch.save(d, logging_folder+'/saved_latent_vecs/latent_vec_' + str(random_seed) + '.pt')

            if epsilon * alpha_d.mean().item() < epsilon*alpha_d_new.mean().item():
                d = d_new

    time2 = time.time()
    print('training-time: ', time2 - time1)

if __name__ == "__main__":
    run_training(parser_args.device, batch_size, maxLength, n_iters, target_feature_index, parser_args.sampleRadius)


