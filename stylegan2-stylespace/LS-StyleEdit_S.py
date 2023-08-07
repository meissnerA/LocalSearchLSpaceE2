import sys
import os
import argparse
import click
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
import legacy
import types
from training.networks import SynthesisNetwork,SynthesisBlock,SynthesisLayer,ToRGBLayer

network_pkl='../pretrained_models/ffhq.pkl'
stylegan2_path = "./"
size = 1024
noise_mode = 'const'
truncation_psi = 0.5
n_iters = 20001

parser = argparse.ArgumentParser(description="optimizing attribute vector for StyleGAN2")
parser.add_argument("device", metavar="device", type=str, help="enter torch device")
parser.add_argument("sampleRadius", metavar="sampleRadius", type=float, help="scales the amount of noise used in the local search")
parser.add_argument("target_feature_index", metavar="target_feature_index", type=int, help="index of a target attribute according to Celeba, 31=Smiling, 9=Hair color")
parser.add_argument("maxLength", metavar="maxLength", type=str, help="the maximum Length a attribute vector before we normalize, usually values < 20 work fine")
parser.add_argument("batch_size", metavar="batch_size", type=str, help="how many samples to average over, 1-8 worked all fine in our experiments")
parser.add_argument("output_path", metavar="output_path", type=str, help="path to save the attribute vectors to e.g. /training_runs/stylegan2")

parser_args = parser.parse_args()
device = parser_args.device

def LoadModel(network_pkl,device):
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    
    G.synthesis.forward=types.MethodType(SynthesisNetwork.forward,G.synthesis)
    G.synthesis.W2S=types.MethodType(SynthesisNetwork.W2S,G.synthesis)
    
    for res in G.synthesis.block_resolutions:
        block = getattr(G.synthesis, f'b{res}')
        # print(block)
        block.forward=types.MethodType(SynthesisBlock.forward,block)
        
        if res!=4:
            layer=block.conv0
            layer.forward=types.MethodType(SynthesisLayer.forward,layer)
            layer.name='conv0_resolution_'+str(res)
            
        layer=block.conv1
        layer.forward=types.MethodType(SynthesisLayer.forward,layer)
        layer.name='conv1_resolution_'+str(res)
        
        layer=block.torgb
        layer.forward=types.MethodType(ToRGBLayer.forward,layer)
        layer.name='toRGB_resolution_'+str(res)
        
    
    return G

G = LoadModel(network_pkl, parser_args.device)   
label = torch.zeros([1, G.c_dim], device=parser_args.device)

file_to_read = open("../pretrained_models/resnet_092_all_attr_5_epochs.pkl", "rb")
regressor = pickle.load(file_to_read)
file_to_read.close()
regressor = regressor.to(device)
regressor.eval()

def add_attr_vec(encoded_styles, attr_vec, epsilon):
    new_attr_vec = {}
    i=0
    for key in encoded_styles.keys():
        new_attr_vec[key] = encoded_styles[key] + attr_vec[0:, i:i+encoded_styles[key].shape[1]]*epsilon
        i+=encoded_styles[key].shape[1]
    return new_attr_vec

z = torch.from_numpy(np.random.RandomState(0).randn(1, G.z_dim)).to(device)
w0 = G.mapping(z,label, truncation_psi=truncation_psi)
s=G.synthesis.W2S(w0)
d0 = torch.zeros(1, 9088)

i=0
for key in s.keys():
    d0[0:, i:i+s[key].shape[1]] = s[key] 
    i+=s[key].shape[1]

for random_seed in range(1, 1000):
    z = torch.from_numpy(np.random.RandomState(random_seed).randn(1, G.z_dim)).to(device)
    w = G.mapping(z,label, truncation_psi=truncation_psi)
    w0 = torch.cat([w0, w], axis=0)
    s=G.synthesis.W2S(w)
    d = torch.zeros(1, 9088)
    i=0
    for key in s.keys():
        d[0:, i:i+s[key].shape[1]] = s[key] 
        i+=s[key].shape[1]
        
    d0 = torch.cat([d0, d], axis=0)
feature_normalization = d0.std(axis=0).to(parser_args.device)

def permutate_attr_vec(attr_vec, sampleRadius, batch_size, random_seed):    
    noise = sampleRadius * torch.from_numpy(np.random.RandomState(random_seed).randn(batch_size, 9088)).to(parser_args.device)
    noise = noise*feature_normalization
    return attr_vec + noise


def run_training(device, batch_size, maxLength, n_iters, target_feature_index, sampleRadius):
    logging_folder = stylegan2_path + output_path + f'/our_approach_stylespace_feature_{target_feature_index}_maxLenght_{maxLength}_lr_{sampleRadius}_batch_size{batch_size}'
    if not os.path.exists(logging_folder):
        os.makedirs(logging_folder)

    if not os.path.exists(logging_folder + '/saved_latent_vecs'):
        os.makedirs(logging_folder + '/saved_latent_vecs')

    with open(logging_folder + "/training_log.txt", "a") as myfile:
                myfile.write(f'./Hyperparameter stylespace: feature_{target_feature_index} maxLenght_{maxLength} sampleRadius_{sampleRadius} batch_size{batch_size}' + '\n')

    d = torch.zeros(1, 9088).to(device) # attribute vector    
    for random_seed in range(n_iters):
        with torch.no_grad():
            z = torch.from_numpy(np.random.RandomState(random_seed).randn(batch_size, 512)).to(device)
            w = G.mapping(z, label, truncation_psi=truncation_psi)
            encoded_styles=G.synthesis.W2S(w)

            epsilon = np.random.choice([-1,1])
            styles_d = add_attr_vec(encoded_styles, d, epsilon)

            img_d = G.synthesis(None, encoded_styles=styles_d,noise_mode='const')
            img_d = F.interpolate(img_d, size=256)
            alpha_d = regressor(img_d)[:, target_feature_index]

            d_new = permutate_attr_vec(d, sampleRadius, batch_size, random_seed)
            if torch.norm(d_new/feature_normalization).item() > maxLength:
                d_new = maxLength*d_new/torch.norm(d_new/feature_normalization)

            styles_d_new = add_attr_vec(encoded_styles, d_new, epsilon)
            img_d_new = G.synthesis(None, encoded_styles=styles_d_new,noise_mode='const')

            img_d_new = F.interpolate(img_d_new, size=256)
            alpha_d_new = regressor(img_d_new)[:, target_feature_index]

            with open(logging_folder + "/training_log.txt", "a") as myfile:
                myfile.write('pred_old: ' + str(alpha_d.mean().item()) + 'pred_new: ' + str(alpha_d_new.mean().item()) + '\n')

            if random_seed%int(1000) == 0:
                torch.save(d, logging_folder+'/saved_latent_vecs/latent_vec_' + str(random_seed) + '.pt')

            if epsilon * alpha_d.mean().item() < epsilon*alpha_d_new.mean().item():
                d = d_new

                
if __name__ == "__main__":
    run_training(parser_args.device, batch_size, maxLength, n_iters, target_feature_index, parser_args.sampleRadius)
