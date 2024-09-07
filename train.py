"""
Author: Gao Yu
Company: Bosch Research / Asia Pacific
Date: 2024-08-03
Description: training script for stegaformer
"""

import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as tv_utils
import lpips
from torch.cuda import amp
from stegaformer import Encoder, Decoder
from utils import rgb_to_yuv, get_message_accuracy, MIMData, InfiniteSamplerWrapper
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import argparse

parser = argparse.ArgumentParser(description='Params for stf model')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--bpp', type=int, default=1)
parser.add_argument('--query', type=str, default='im')
parser.add_argument('--iterations', type=int, default=80000)
parser.add_argument('--base_lr', type=float, default=5e-4)

parser.add_argument('--win_size', type=int, default=16)
parser.add_argument('--model_path', type=str, default='./stega_model/')
parser.add_argument('--data_path', type=str, default='./stega_data')
parser.add_argument('--dataset', type=str, default='div2k')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--ext', type=str, default='_255_w')
parser.add_argument('--n_clamp', type=int, default=4)
parser.add_argument('--msg_scale', type=int, default=1)
parser.add_argument('--enable_img_norm', action='store_true', help='Image Normalize to 0 ~ 1 if True')
parser.add_argument('--msg_range', type=int, default=1)
parser.add_argument('--msg_pose', type=str, default='learn')

args = parser.parse_args()

message_L = 16*args.bpp
message_N = 64*64
im_size = 256
device_id = args.device

cal_psnr = PeakSignalNoiseRatio().cuda(device_id)
cal_ssim = StructuralSimilarityIndexMeasure().cuda(device_id)

def ssim(img1, img2):
    s = cal_ssim(img1, img2)
    return s.cpu().detach().numpy()

def psnr(cover, generated):
    psnr = cal_psnr(generated, cover)
    return psnr.cpu().detach().numpy()

def compute_image_score(cover, generated):
    
    p = psnr(cover, generated)
    s = ssim(cover, generated)
    
    return p, s


def test(test_loader, encoder, decoder, message_N):
    encoder.eval()
    decoder.eval()
    dataiter = iter(test_loader)
    with torch.no_grad():
        acc = []
        psnr = []
        ssim = []
        for i in range(test_loader.__len__()):
            images,messages = next(dataiter)
            messages = messages.cuda(device_id)
            images = images.cuda(device_id)
            
            enco_images = encoder(images, messages)
            enco_images = torch.clamp(enco_images,0,255)
            
            deco_messages = decoder(enco_images)
            #please sigmoid first for the decoded message
            if msg_range == 1:
                deco_messages = torch.sigmoid(deco_messages)
            ac = get_message_accuracy(messages, deco_messages, message_N)
            p, s = compute_image_score(images, enco_images)
            
            acc.append(ac)
            psnr.append(p)
            ssim.append(s)
    
    encoder.train()
    decoder.train()
    
    return sum(acc)/len(acc), sum(psnr)/len(psnr), sum(ssim)/len(ssim)

train_path = os.path.join(args.data_path,args.dataset,'train')
test_path = os.path.join(args.data_path,args.dataset,'val')

msg_range = args.msg_range
norm = args.enable_img_norm
train_dataset = MIMData(data_path=train_path, num_message=message_N, message_size=message_L, image_size=(im_size, im_size),
                        dataset=args.dataset, msg_r=msg_range)
test_dataset = MIMData(data_path=test_path, num_message=message_N, message_size=message_L, image_size=(im_size, im_size),
                       dataset=args.dataset, msg_r=msg_range)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, sampler=InfiniteSamplerWrapper(train_dataset), shuffle=False, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

image_iter = iter(train_dataloader)

scale = args.msg_scale
encoder_eb_dim = int(message_L*scale)
decoder_eb_dim = int(message_L*scale)



query_type = args.query
encoder = Encoder(msg_L=message_L, embed_dim=encoder_eb_dim, Q=query_type, win_size=args.win_size, msg_pose=args.msg_pose)
decoder = Decoder(img_size=256, msg_L=message_L, embed_dim=decoder_eb_dim, win_size=args.win_size, msg_pose=args.msg_pose)

sf_lr = args.base_lr
clamper = args.n_clamp

encoder_optimizer = optim.Adam(encoder.parameters(), lr=sf_lr,weight_decay=1e-5)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=sf_lr,weight_decay=1e-5)

encoder.cuda(device_id)
decoder.cuda(device_id)

max_iter = args.iterations
encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, int(max_iter/clamper), gamma=0.5)
decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, int(max_iter/clamper), gamma=0.5)

save_path = os.path.join(args.model_path,str(args.msg_range)+'_'+str(args.bpp)+args.ext+str(args.win_size)+'_'+str(args.msg_pose)+'_'+str(args.query),args.dataset,'model')
isExist = os.path.exists(save_path)
if not isExist:
    os.makedirs(save_path)

log_path = os.path.join(args.model_path,str(args.msg_range)+'_'+str(args.bpp)+args.ext+str(args.win_size)+'_'+str(args.msg_pose)+'_'+str(args.query),args.dataset,'logs')
isExist = os.path.exists(log_path)
if not isExist:
    os.makedirs(log_path)
    
log_img_path = os.path.join(args.model_path,str(args.msg_range)+'_'+str(args.bpp)+args.ext+str(args.win_size)+'_'+str(args.msg_pose)+'_'+str(args.query),args.dataset,'samples')
isExist = os.path.exists(log_img_path)
if not isExist:
    os.makedirs(log_img_path)
    
writer = SummaryWriter(log_dir=log_path)

img_criteria = nn.MSELoss().cuda(device_id)
if msg_range > 1:
    msg_criteria = nn.MSELoss().cuda(device_id)
else:
    msg_criteria = nn.BCEWithLogitsLoss().cuda()

lpips_alex = lpips.LPIPS(net="vgg", verbose=False)
lpips_alex.cuda(device_id)

save_log_interval = 50
save_image_interval = 4000
save_model_interval = int(max_iter/2)
test_interval = 500

image_loss_scale = 1.0
image_loss_ramp = int(max_iter/2)
secret_loss_scale = 100
secret_loss_ramp = 1
lpips_loss_scale = 0.1
lpips_loss_ramp = int(max_iter/2)

max_psnr = 0.0
max_ssim = 0.0
max_acc = 0.0
scaler = amp.GradScaler()

#for k in tqdm(range(max_iter)):
for k in range(max_iter):
    
    s_im_loss = min(image_loss_scale * k / image_loss_ramp, image_loss_scale)
    s_msg_loss = min(secret_loss_scale * k / secret_loss_ramp, secret_loss_scale)
    s_lpips_loss = min(lpips_loss_scale * k / lpips_loss_ramp, lpips_loss_scale)

    images, messages = next(image_iter)
    
    encoder_optimizer.zero_grad(set_to_none=True)
    decoder_optimizer.zero_grad(set_to_none=True)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        messages = messages
        messages = messages.cuda(device_id)
        images = images.cuda(device_id)

        enco_images = encoder(images, messages)
        enco_images = torch.clamp(enco_images,0,255)
    
        deco_messages = decoder(enco_images)

        message_loss = msg_criteria(deco_messages, messages)
        image_loss = img_criteria(enco_images, images)
        # the images must be normalized to compute lpips
        normalized_images = images/255. * 2 - 1.
        normalized_enco_images = enco_images/255. * 2 - 1.
        lpips_loss = torch.mean(lpips_alex(normalized_images, normalized_enco_images))
        
        loss = s_im_loss*image_loss + s_msg_loss*message_loss + s_lpips_loss*lpips_loss
    
    scaler.scale(loss).backward()
    scaler.step(encoder_optimizer)
    scaler.step(decoder_optimizer)
    scaler.update()
    encoder_scheduler.step()
    decoder_scheduler.step()
    
    with torch.no_grad(): 
        if (k+1) % save_log_interval == 0 or (k + 1) == max_iter:
            messages = messages.float()
            #please sigmoid first for the decoded message
            if msg_range == 1:
                 deco_messages = torch.sigmoid(deco_messages)
            deco_messages = deco_messages.float()

            train_message_acc = get_message_accuracy(messages, deco_messages, message_N)
            
            train_psnr, train_ssim = compute_image_score(images, enco_images)
            
            writer.add_scalar('Train Image Loss', image_loss.item(), k + 1)
            writer.add_scalar('Train LPIPs Loss', lpips_loss.mean(), k + 1)
            writer.add_scalar('Train Message Loss', message_loss.item(), k + 1)
            writer.add_scalar('Train Message Acc', train_message_acc.item(), k + 1)
            writer.add_scalar('Train PSNR', train_psnr, k + 1)
            writer.add_scalar('Train SSIM', train_ssim, k + 1)
            
        if (k+1) % test_interval == 0 and (k+1) > save_model_interval:
            test_message_acc, test_psnr, test_ssim = test(test_dataloader, encoder, decoder, message_N)
            
            writer.add_scalar('Test Message Acc', test_message_acc.item(), k + 1)
            writer.add_scalar('Test PSNR', test_psnr, k + 1)
            writer.add_scalar('Test SSIM', test_ssim, k + 1)
            
            if (k+1) > save_model_interval:
                if test_psnr > max_psnr:
                    max_psnr = test_psnr
                    state_dict = decoder.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device('cpu'))
                        torch.save(state_dict, save_path+'/best_psnr_decoder.pth.tar')
                    state_dict = encoder.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device('cpu'))
                    torch.save(state_dict, save_path+'/best_psnr_encoder.pth.tar')   
                    
                if test_ssim > max_ssim:
                    max_ssim = test_ssim
                    state_dict = decoder.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device('cpu'))
                    torch.save(state_dict, save_path+'/best_ssim_decoder.pth.tar')

                    state_dict = encoder.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device('cpu'))
                    torch.save(state_dict, save_path+'/best_ssim_encoder.pth.tar')
                    
                if test_message_acc > max_acc:
                    max_acc = test_message_acc
                    state_dict = decoder.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device('cpu'))
                    torch.save(state_dict, save_path+'/best_acc_decoder.pth.tar')

                    state_dict = encoder.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device('cpu'))
                    torch.save(state_dict, save_path+'/best_acc_encoder.pth.tar')
            
        if (k + 1) % save_image_interval == 0 or (k + 1) == max_iter:
            save_images = enco_images.cpu()
            tv_utils.save_image(save_images,f'{log_img_path}/{str(k + 1).zfill(6)}.png',nrow=args.batch_size,normalize=True,range=(0, 1))
            save_images = images.cpu()
            tv_utils.save_image(save_images,f'{log_img_path}/{str(k + 1).zfill(6)}_cover.png',nrow=args.batch_size,normalize=True,range=(0, 1))
writer.close()