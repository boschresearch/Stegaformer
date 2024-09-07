"""
Author: Gao Yu
Company: Bosch Research / Asia Pacific
Date: 2024-08-03
Description: main functions for stegaformer
"""

import numpy as np
import torch
import torch.nn as nn
import time

from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import get_init_weights_vit
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D
from modules import Downsample, Upsample, Msg_Block, InputProj, BasicLayer, BasicLayerM

# Encoder for Stegaformer

class Encoder(nn.Module):
    def __init__(self, img_size=256, dd_in=3, msg_L=16, Q='im',
                 embed_dim=32, depths=[1, 1, 1, 1, 1, 1, 1], num_heads=[2, 2, 2, 2, 2, 2, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True,drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, patch_norm=True,token_projection='linear', token_mlp='leff',
                 dowsample=Downsample, upsample=Upsample, msg_pose='learn'):
        super().__init__()
        
        # Message dim should >= the length of message segment
        self.msg_dim = embed_dim
        # Select the type of query
        self.q_select = Q
        # Select the type of positional encoding
        self.m_pose = msg_pose
        

        layers = []
        for i_layer in range(len(depths)//2-1):
            layers += [nn.Sequential(nn.Conv2d(msg_L, embed_dim*(2 ** (i_layer+2)), kernel_size=1, stride=1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(embed_dim*(2 ** (i_layer+2)), embed_dim*2*(2 ** (i_layer+1)), kernel_size=1+2*i_layer, stride=int(2**i_layer), padding=i_layer),
                                    nn.LeakyReLU(inplace=True))
                    ]
        
        layers += [nn.Sequential(nn.Conv2d(msg_L, embed_dim*8, kernel_size=1, stride=1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(embed_dim*8, embed_dim*8, kernel_size=1, stride=1),
                                    nn.LeakyReLU(inplace=True))
                    ]
        
        self.msg_layers = nn.Sequential(*layers)
        
        layers = []
        for i_layer in range(len(depths)//2-1):
            layers += [Msg_Block(dim=embed_dim*2*(2 ** (i_layer+1)), num_heads=num_heads[i_layer+1])]
        layers += [Msg_Block(dim=embed_dim*8, num_heads=num_heads[4])]
        self.msg_att = nn.Sequential(*layers)
        
        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        self.encoderlayer_0 = nn.Sequential(
                                            nn.Conv2d(dd_in, embed_dim, kernel_size=3, stride=1, padding=1),
                                            nn.LeakyReLU(inplace=True)
                                            )
        
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        
        self.encoderlayer_1 = nn.Sequential(
                                            nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=3, stride=1, padding=1),
                                            nn.LeakyReLU(inplace=True)
                                            )
        
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        
        self.encoderlayer_2 = BasicLayerM(dim=embed_dim*4, input_resolution=(img_size // 4,img_size // 4), depth=depths[2], num_heads=num_heads[2], window_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
                                          drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, downsample=None)
        
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)
        
        # Bottleneck
        
        self.conv = BasicLayerM(dim=embed_dim*8, input_resolution=(img_size // 8,img_size // 8), depth=depths[3], num_heads=num_heads[3], window_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, downsample=None)
        
        self.upsample_0 = upsample(embed_dim*8, embed_dim*4)
        
        self.decoderlayer_0 = BasicLayerM(dim=embed_dim*(4+4), input_resolution=(img_size // 4,img_size // 4), depth=depths[4], num_heads=num_heads[4], window_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
                                         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, downsample=None)
        
        self.upsample_1 = upsample(embed_dim*8, embed_dim*4)
        
        self.decoderlayer_1 = nn.Sequential(
                                            nn.Conv2d(embed_dim*(4+2), embed_dim*(4+2), kernel_size=3, stride=1, padding=1),
                                            nn.LeakyReLU(inplace=True)
                                            )
        
        self.upsample_2 = upsample(embed_dim*6, embed_dim*3)
        
        self.decoderlayer_2 = nn.Conv2d(embed_dim*(3+1), dd_in, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)
        
        layers = []
        for i_layer in range(len(depths)//2-1):
            layers += [norm_layer(embed_dim*2*(2 ** (i_layer+1)))]
        layers += [norm_layer(embed_dim*8)]
        self.norm = nn.Sequential(*layers)
        
        self.conv_norm1 = nn.LayerNorm(embed_dim)
        self.conv_norm2 = nn.LayerNorm(embed_dim*2)
        self.conv_norm3 = nn.LayerNorm(embed_dim*6)
        
        if self.m_pose == 'learn':
            self.msg_pos_embed_l0 = nn.Parameter(torch.randn(1, int(64*64), embed_dim*4) * .02)
            self.msg_pos_embed_l1 = nn.Parameter(torch.randn(1, int(32*32), embed_dim*8) * .02)
            self.msg_pos_embed_l2 = nn.Parameter(torch.randn(1, int(64*64), embed_dim*8) * .02)
            trunc_normal_(self.msg_pos_embed_l0, std=.02)
            trunc_normal_(self.msg_pos_embed_l1, std=.02)
            trunc_normal_(self.msg_pos_embed_l2, std=.02)        
        elif self.m_pose == '1d':
            self.msg_pos_embed_l0 =  PositionalEncoding1D(embed_dim*4)
            self.msg_pos_embed_l1 =  PositionalEncoding1D(embed_dim*8)
            self.msg_pos_embed_l2 =  PositionalEncoding1D(embed_dim*8)
        elif self.m_pose == '2d':
            self.msg_pos_embed_l0 =  PositionalEncoding2D(embed_dim*4)
            self.msg_pos_embed_l1 =  PositionalEncoding2D(embed_dim*8)
            self.msg_pos_embed_l2 =  PositionalEncoding2D(embed_dim*8)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, m, mask=None):
        conv0 = self.encoderlayer_0(x)
        
        b,d,h,w = conv0.shape
        conv0 = conv0.view(b,d,h*w)
        conv0 = conv0.permute(0, 2, 1)
        conv0 = self.conv_norm1(conv0)
        pool0 = self.dowsample_0(conv0)
        
        b,n,d = pool0.shape
        pool0 = pool0.permute(0, 2, 1)
        pool0 = pool0.view(b, d, int(np.sqrt(n)), int(np.sqrt(n)))
        conv1 = self.encoderlayer_1(pool0)

        b,d,h,w = conv1.shape
        conv1 = conv1.view(b,d,h*w)
        conv1 = conv1.permute(0, 2, 1)
        conv1 = self.conv_norm2(conv1)
        pool1 = self.dowsample_1(conv1)
        
        # Msg Projection
        b,n1,_ = pool1.shape
        b,n2,c2 = m.shape
        msg_eb = m.view(b, int(np.sqrt(n1)), int(np.sqrt(n1)), c2)
        msg_eb = torch.permute(msg_eb, (0, 3, 1, 2))
        
        msg_eb = self.msg_layers[0](msg_eb)
        msg_eb = msg_eb.flatten(2)
        msg_eb = msg_eb.permute(0, 2, 1)
        # Msg Attention
        
        if self.m_pose == 'learn':
            msg_eb = self.msg_att[0](msg_eb + self.msg_pos_embed_l0)
        elif self.m_pose == '1d':
            msg_eb = self.msg_att[0](msg_eb + self.msg_pos_embed_l0(msg_eb))
        elif self.m_pose == '2d':
            b,n,d = msg_eb.shape
            msg_eb = msg_eb.view(b, int(np.sqrt(n)), int(np.sqrt(n)), d)
            msg_eb = msg_eb + self.msg_pos_embed_l0(msg_eb)
            msg_eb = msg_eb.view(b, n, d)
            msg_eb = self.msg_att[0](msg_eb)
        else:
            msg_eb = self.msg_att[0](msg_eb)
            
        msg_eb = self.norm[0](msg_eb)
        
        # Im and Msg encode
        if self.q_select == 'im':
            conv2 = self.encoderlayer_2(pool1+msg_eb, pool1)
            pool2 = self.dowsample_2(conv2)
        elif self.q_select == 'msg':
            conv2 = self.encoderlayer_2(pool1+msg_eb, msg_eb)
            pool2 = self.dowsample_2(conv2)
        elif self.q_select == 'im+msg':
            conv2 = self.encoderlayer_2(pool1+msg_eb, pool1+msg_eb)
            pool2 = self.dowsample_2(conv2)
        
        # Msg Projection
        b,n1,_ = pool1.shape
        b,_,c2 = m.shape
        msg_eb = m.view(b, int(np.sqrt(n1)), int(np.sqrt(n1)), c2)
        msg_eb = torch.permute(msg_eb, (0, 3, 1, 2))
        msg_eb = self.msg_layers[1](msg_eb)
        msg_eb = msg_eb.flatten(2)
        msg_eb = msg_eb.permute(0, 2, 1)
        # Msg Attention
        if self.m_pose == 'learn':
            msg_eb = self.msg_att[1](msg_eb + self.msg_pos_embed_l1)
        elif self.m_pose == '1d':
            msg_eb = self.msg_att[1](msg_eb + self.msg_pos_embed_l1(msg_eb))
        elif self.m_pose == '2d':
            b,n,d = msg_eb.shape
            msg_eb = msg_eb.view(b, int(np.sqrt(n)), int(np.sqrt(n)), d)
            msg_eb = msg_eb + self.msg_pos_embed_l1(msg_eb)
            msg_eb = msg_eb.view(b, n, d)
            msg_eb = self.msg_att[1](msg_eb)
        else:
            msg_eb = self.msg_att[1](msg_eb)
            
        msg_eb = self.norm[1](msg_eb)

        # Im and Msg encode
        # Bottleneck
        if self.q_select == 'im':
            conv3 = self.conv(pool2+msg_eb, pool2)
        elif self.q_select == 'msg':
            conv3 = self.conv(pool2+msg_eb, msg_eb)
        elif self.q_select == 'im+msg':
            conv3 = self.conv(pool2+msg_eb, pool2+msg_eb)        
        # Im Decode and Reconstruction
        up0 = self.upsample_0(conv3)
        
        # Msg Projection
        b,n1,_ = pool1.shape
        b,_,c2 = m.shape
        msg_eb = m.view(b, int(np.sqrt(n1)), int(np.sqrt(n1)), c2)
        msg_eb = torch.permute(msg_eb, (0, 3, 1, 2))
        msg_eb = self.msg_layers[2](msg_eb)
        msg_eb = msg_eb.flatten(2)
        msg_eb = msg_eb.permute(0, 2, 1)
        # Msg Attention
        if self.m_pose == 'learn':
            msg_eb = self.msg_att[2](msg_eb + self.msg_pos_embed_l2)
        elif self.m_pose == '1d':
            msg_eb = self.msg_att[2](msg_eb + self.msg_pos_embed_l2(msg_eb))
        elif self.m_pose == '2d':
            b,n,d = msg_eb.shape
            msg_eb = msg_eb.view(b, int(np.sqrt(n)), int(np.sqrt(n)), d)
            msg_eb = msg_eb + self.msg_pos_embed_l2(msg_eb)
            msg_eb = msg_eb.view(b, n, d)
            msg_eb = self.msg_att[2](msg_eb)
        else:
            msg_eb = self.msg_att[2](msg_eb)
        msg_eb = self.norm[2](msg_eb)

        # Im and Msg encode
        # Bottleneck
        if self.q_select == 'im':
            deconv0 = self.decoderlayer_0(torch.cat([up0,conv2],-1)+msg_eb, torch.cat([up0,conv2],-1))
        elif self.q_select == 'msg':
            deconv0 = self.decoderlayer_0(torch.cat([up0,conv2],-1)+msg_eb, msg_eb)
        elif self.q_select == 'im+msg':
            deconv0 = self.decoderlayer_0(torch.cat([up0,conv2],-1)+msg_eb, torch.cat([up0,conv2],-1)+msg_eb)

        up1 = self.upsample_1(deconv0)
        
        up1 = torch.cat([up1,conv1],-1)
        b,n,d = up1.shape
        up1 = up1.permute(0, 2, 1)
        up1 = up1.view(b, d, int(np.sqrt(n)), int(np.sqrt(n)))
        
        deconv1 = self.decoderlayer_1(up1)

        b,d,h,w = deconv1.shape
        deconv1 = deconv1.view(b,d,h*w)
        deconv1 = deconv1.permute(0, 2, 1)
        deconv1 = self.conv_norm3(deconv1)
        up2 = self.upsample_2(deconv1)
        
        up2 = torch.cat([up2,conv0],-1)
        b,n,d = up2.shape
        up2 = up2.permute(0, 2, 1)
        up2 = up2.view(b, d, int(np.sqrt(n)), int(np.sqrt(n)))
        
        y = self.decoderlayer_2(up2)
        
        return x + y

# Decoder for Stegaformer
class Decoder(nn.Module):
    def __init__(self, img_size=256, in_chans=3, msg_L=96,
                 embed_dim=128, depths=(1, 1, 1, 1), num_heads=(2, 2, 2, 2),
                 win_size=16, mlp_ratio=4., qkv_bias=True,drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,weight_init='',msg_pose='learn'):
        super().__init__()
        
        self.msg_length = msg_L
        self.msg_dim = embed_dim
        
        self.m_pose = msg_pose
        
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=1, stride=1, act_layer=nn.LeakyReLU)
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        layers = []
        
        layers += [BasicLayer(
            dim=int(embed_dim),
            input_resolution=(img_size, img_size),
            depth=depths[self.num_layers-4],
            num_heads=num_heads[self.num_layers-4],
            window_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0.,
            norm_layer=norm_layer,
            downsample=None)
                    ]
        
        layers += [BasicLayer(
            dim=int(embed_dim*2),
            input_resolution=(img_size//2, img_size//2),
            depth=depths[self.num_layers-3],
            num_heads=num_heads[self.num_layers-3],
            window_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0.,
            norm_layer=norm_layer,
            downsample=None)
                    ]
            
        layers += [BasicLayerM(
                dim=int(embed_dim*4),
                input_resolution=(img_size//4, img_size//4),
                depth=depths[self.num_layers-2],
                num_heads=num_heads[self.num_layers-2],
                window_size=win_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None)
                       ]

        layers += [BasicLayerM(
                dim=embed_dim*4,
                input_resolution=(img_size // 4,img_size // 4),
                depth=depths[self.num_layers-1],
                num_heads=num_heads[self.num_layers-1],
                window_size=win_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None)
                            ]
        
        self.layers = nn.Sequential(*layers)
        
        self.query_head1 = nn.Linear(self.num_features*4, self.num_features*4)
        self.query_head2 = nn.Linear(self.num_features*4, self.num_features*4)
        
        self.query_attn1 = Msg_Block(dim=self.num_features*4, num_heads=4)
        self.query_attn2 = Msg_Block(dim=self.num_features*4, num_heads=4)
        
        self.msg_head = nn.Linear(self.num_features*4, self.msg_length)
        self.msg_attn = Msg_Block(dim=self.num_features*4, num_heads=4)
        
        self.q_norm1 = norm_layer(self.num_features*4)
        self.q_norm2 = norm_layer(self.num_features*4)
        self.msg_norm = norm_layer(self.num_features*4)
        
        if self.m_pose == 'learn':
            self.msg_pos = nn.Parameter(torch.randn(1, int(64*64), embed_dim*4) * .02)
            trunc_normal_(self.msg_pos, std=.02)
        elif self.m_pose == '1d':
            self.msg_pos =  PositionalEncoding1D(embed_dim*4)
        elif self.m_pose == '2d':
            self.msg_pos =  PositionalEncoding2D(embed_dim*4)
        
        self.dowsample_0 = Downsample(embed_dim, embed_dim*2)
        self.dowsample_1 = Downsample(embed_dim*2, embed_dim*4)
        
        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        named_apply(get_init_weights_vit(mode, head_bias=0.), self)

    def decode(self, x, pm=False):
        # Feature Extraction
        x = self.layers[0](x)
        x = self.dowsample_0(x)
        x = self.layers[1](x)
        x = self.dowsample_1(x)
                    
        if self.m_pose == 'learn':
            query1 = self.query_attn1(x + self.msg_pos)
        elif self.m_pose == '1d':
            query1 = self.query_attn1(x + self.msg_pos(x))
        elif self.m_pose == '2d':
            b,n,d = x.shape
            x = x.view(b, int(np.sqrt(n)), int(np.sqrt(n)), d)
            x = x + self.msg_pos(x)
            x = x.view(b, n, d)
            query1 = self.query_attn1(x)
        else:
            query1 = self.query_attn1(x)
            
        query1 = self.q_norm2(query1)
        
        query1 = self.query_head1(query1)
        
        x = self.layers[2](x,query1)
        # learn q stage 2
        if self.m_pose == 'learn':
            query2 = self.query_attn2(x + self.msg_pos)
        elif self.m_pose == '1d':
            query2 = self.query_attn2(x + self.msg_pos(x))
        elif self.m_pose == '2d':
            b,n,d = x.shape
            x = x.view(b, int(np.sqrt(n)), int(np.sqrt(n)), d)
            x = x + self.msg_pos(x)
            x = x.view(b, n, d)
            query2 = self.query_attn2(x)
        else:
            query2 = self.query_attn2(x)
            
        query2 = self.q_norm2(query2)
        
        query2 = self.query_head2(query2)
        # message embed
        msg_eb = self.layers[3](x,query2)
        # Msg Attention
        if self.m_pose == 'learn':
            msg_eb = self.msg_attn(msg_eb + self.msg_pos)
        elif self.m_pose == '1d':
            msg_eb = self.msg_attn(x + self.msg_pos(x))
        elif self.m_pose == '2d':
            b,n,d = x.shape
            x = x.view(b, int(np.sqrt(n)), int(np.sqrt(n)), d)
            x = x + self.msg_pos(x)
            x = x.view(b, n, d)
            msg_eb = self.msg_attn(x)
        else:
            msg_eb = self.msg_attn(x)
             
        msg_eb = self.msg_norm(msg_eb)
        # Decode Rough Msg
        msg = self.msg_head(msg_eb)
        
        return msg

    def forward(self, x, pm=False):
        x = self.input_proj(x)
        x = self.decode(x,pm)
        
        return x
    
if __name__ == '__main__':
    
    bpp = 3
    # set length of message segment 
    message_L = 16*bpp
    # set number of message segments
    message_N = 64*64
    # set the scale of message embedding usually >= message_L
    scale = 2
    eb_dim = int(message_L*scale)

    encoder = Encoder(msg_L=message_L, embed_dim=eb_dim, Q='im', win_size=16, msg_pose='2d').cuda()
    decoder = Decoder(img_size=256, msg_L=message_L, embed_dim=eb_dim, win_size=16, msg_pose='2d').cuda()

    total_running_time = 0
    repetitions = 100
    # test the exeuating time of encoder and decoder
    for _ in range(repetitions):
        im = torch.randn(1,3,256,256).cuda()
        msg = torch.randn(1,64*64,message_L).cuda()
        start_time = time.time()
        stego = encoder(im,msg)
        d_msg = decoder(stego)
        end_time = time.time()
        running_time = end_time - start_time
        total_running_time += running_time

    average_running_time = total_running_time / repetitions
    print("Average running time over", repetitions, "repetitions:", average_running_time, "seconds")