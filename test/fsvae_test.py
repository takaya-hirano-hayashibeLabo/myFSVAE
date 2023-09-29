import sys
from pathlib import Path
ROOT=str(Path(__file__).parent.parent)
sys.path.append(ROOT)

import torch
from torch.utils.data.dataloader import DataLoader
import argparse
import pandas as pd
import os
import cv2
import numpy as np
from copy import deepcopy

from src.network_parser import parse
from src import global_v as glv
from src.fsvae import fsvae


def train(network, trainloader, opti, epoch)->pd.DataFrame:
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']
    
    mean_q_z = 0
    mean_p_z = 0
    mean_sampled_z = 0
    
    loss_table=[]
    loss_column=["epoch","batch_idx","loss","reconst","distance"]

    network = network.train()
    
    for batch_idx, real_img in enumerate(trainloader):   
        opti.zero_grad()
        # real_img = real_img.to(init_device, non_blocking=True)
        # labels = labels.to(init_device, non_blocking=True)
        # direct spike input
        spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)
        x_recon, q_z, p_z, sampled_z = network(spike_input, scheduled=glv.network_config['scheduled']) # sampled_z(B,C,1,1,T)
        
        if glv.network_config['loss_func'] == 'mmd':
            losses = network.loss_function_mmd(real_img, x_recon, q_z, p_z)
        elif glv.network_config['loss_func'] == 'kld':
            losses = network.loss_function_kld(real_img, x_recon, q_z, p_z)
        else:
            raise ValueError('unrecognized loss function')
        
        losses['loss'].backward()
        
        opti.step()
        network.weight_clipper()

        # loss_meter.update(losses['loss'].detach().cpu().item())
        # recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
        # dist_meter.update(losses['Distance_Loss'].detach().cpu().item())
        loss_table_idx=[epoch,batch_idx]+[loss.detach().cpu().item() for loss in losses.values()]
        loss_table=loss_table+[loss_table_idx]
        # print(loss_table)
        loss_pd=pd.DataFrame(loss_table,columns=loss_column)

        mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx+1) # (C,k,T)
        mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx+1) # (C,k,T)
        mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (batch_idx+1) # (C,T)

        print(f'Train[{epoch}/{max_epoch}] [{batch_idx}/{len(trainloader)}] Loss: {loss_pd["loss"].mean()}, RECONS: {loss_pd["reconst"].mean()}, DISTANCE: {loss_pd["distance"].mean()}')

        # if batch_idx == len(trainloader)-1:
        #     os.makedirs(f'checkpoint/{args.name}/imgs/train/', exist_ok=True)
        #     torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_input.png')
        #     torchvision.utils.save_image((x_recon+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_recons.png')
        #     writer.add_images('Train/input_img', (real_img+1)/2, epoch)
        #     writer.add_images('Train/recons_img', (x_recon+1)/2, epoch)    

    mean_q_z = mean_q_z.permute(1,0,2) # (k,C,T)
    mean_p_z = mean_p_z.permute(1,0,2) # (k,C,T)

    return loss_pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--config', action='store', dest='config', help='The path of config file')
    # parser.add_argument('--checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--is_train',type=bool,default=False)
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)
    
    device=args.device
    params = parse(args.config)
    network_config = params['Network']
    print(network_config)
    glv.init(network_config, [device])
    
    
    net = fsvae.FSVAE()
    optimizer = torch.optim.Adam(net.parameters(), 
                                lr=glv.network_config['lr'], 
                                betas=(0.9, 0.999), 
                                weight_decay=0.001)
    
    
    # >> 適当な入力テスト >>
    # input_img=torch.rand(size=(50,1,32,32)).unsqueeze(-1).repeat(1, 1, 1, 1, glv.network_config['n_steps'])
    # out=net(input_img, scheduled=glv.network_config['scheduled'])
    # print(out)
    # << 適当な入力テスト <<
    
    
    # >> イラストやの読み込み >>
    data_path=f"{ROOT}/test/test_data"
    # data=[
    #     [cv2.resize(cv2.imread(f"{data_path}/{img}",cv2.IMREAD_GRAYSCALE) 
    #                ,(network_config["input_size"],network_config["input_size"])) ]
    #     for img in os.listdir(data_path)
    #     ]
    data=[
        cv2.resize(cv2.imread(f"{data_path}/{img}") 
                   ,(network_config["input_size"],network_config["input_size"]))
        for img in os.listdir(data_path)
        ]
    data_org=deepcopy(data)
    
    for i in range(len(os.listdir(data_path))):
        cv2.imshow("img",data_org[i]/255)
        cv2.waitKey(0)    
        print(np.sum(data_org[i])/255)
    data=torch.Tensor(np.array(data)/255.0) #正規化してTensorにする
    print(data.shape)
    data=torch.permute(data,(0,3,1,2))
    print(data.shape)
    
    # print(np.random.choice(data.flatten(),size=(200)))
    dataloader=DataLoader(
        dataset=torch.Tensor(data), batch_size=network_config["batch_size"],
        shuffle=True, num_workers=3,drop_last=True
        )
    # << イラストやの読み込み <<
    
    result_path=f"{ROOT}/test/result"
    net.load_state_dict(torch.load(f=f"{result_path}/fsvae_test_rgb.pth",map_location=device))
    loss=pd.DataFrame([])
    print(args.is_train)
    if  args.is_train:
        for e in range(glv.network_config['epochs']):
            
            # >> posteriorの事前分布を更新してる？ >>
            if network_config['scheduled']:
                net.update_p(e, glv.network_config['epochs'])
            # << posteriorの事前分布を更新してる？ <<
                
            train_loss_ep:pd.DataFrame = train(net, dataloader, optimizer, e)
            if len(loss)==0:
                loss=train_loss_ep
            else:
                loss=pd.concat([loss, train_loss_ep],axis=0)
            loss.to_csv(f"{result_path}/fsvae_test_loss.csv",index=False)
            
            # test_loss = test(net, test_loader, e)

            # torch.save(net.state_dict(), f'checkpoint/{args.name}/checkpoint.pth')
            # if train_loss < best_loss:
            #     best_loss = train_loss
            #     torch.save(net.state_dict(maplocation=device), f'checkpoint/{args.name}/best.pth')
        torch.save(net.to(device).state_dict(), f'{result_path}/fsvae_test_rgb.pth')
        
    net.eval()
    with torch.no_grad():
        net.load_state_dict(torch.load(f=f"{result_path}/fsvae_test_rgb.pth",map_location=device))
        spike_input = torch.Tensor(data).unsqueeze(-1).repeat(1, 1, 1, 1, network_config["n_steps"]) # (N,C,H,W,T)
        x_recon, q_z, p_z, sampled_z = net(spike_input, scheduled=network_config['scheduled'])
        x_recon=torch.permute(x_recon,(0,2,3,1))
        x_recon=(x_recon.detach().to("cpu").numpy())
    
    for i in range(len(x_recon)):
        print(np.sum(x_recon[i]))
        # print(data_org[i].shape)
        merge_img=cv2.hconcat([
            np.array(data_org[i]/255,dtype=float),
            np.array(x_recon[i],dtype=float)
        ])
        cv2.imshow("img",merge_img)
        cv2.waitKey(0)
    
    # for i in range(network_config["n_steps"]):
    #     cv2.imshow("img",spike_input.to("cpu").numpy()[1,0,:,:,i])
    #     cv2.waitKey(0)
    #     print(i)
        

    
if __name__ == '__main__':
    main()