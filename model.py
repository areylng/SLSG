#coding=gbk
from torchvision.models import resnet18
import torch
import torch.nn as nn
from attention_module import *

class Decoder (nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(Decoder , self).__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(256+256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(128+128+32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )


        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(64+64+32,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(#nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                 nn.BatchNorm2d( 32),
                                 nn.ReLU(inplace=True))

        self.db4 = nn.Sequential(
            nn.Conv2d(32+64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.res_bn_relu = nn.Sequential(nn.BatchNorm2d(24),
                                         nn.ReLU(inplace=True), )
        self.final_out_seg = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
        )
        self.up2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Init()



    def Init(self):
        mo_list = [self.up1,self.up2,self.up3,self.up4, self.db1,self.db2,self.db3,self.db4,self.final_out_seg,self.res_bn_relu,]
        for m in mo_list:
        #for m in self.block_down1.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, forward_out,aggregate1,aggregate2,aggregate3,bn_out_128x128,x,AUX_out0):
        up1 = self.up1(forward_out)
        cat = torch.cat((up1,aggregate3),dim=1)
        db1 = self.db1(cat)#512

        up2 = self.up2(db1)
        AUX_temp =  self.up2x(AUX_out0)
        AUX_temp = self.up2x(AUX_temp)
        cat = torch.cat((up2,aggregate2,AUX_temp),dim=1)
        db2 = self.db2(cat)#288

        up3 = self.up3(db2)#64,128,128
        AUX_temp = self.up2x(AUX_temp)
        cat = torch.cat((up3,aggregate1,AUX_temp),dim=1)
        db3 = self.db3(cat)#160

        x = self.up4(db3)#32,128,128
        x = torch.cat((x, bn_out_128x128), dim=1)
        db4 = self.db4(x)

        out = self.final_out_seg(db4)
        return out,1#AUX


class Encoder(nn.Module):
    def __init__(self, pretrained=True, head_layers=[512,512,512,512,512,512,512,512,128], num_classes=2,
                 data_type=None,use_patch_core=False,encoder_freeze=False,epoch_MODE=0):
        super(Encoder, self).__init__()
        self.epoch_MODE=epoch_MODE
        self.resnet18 = resnet18(pretrained=True)
        try:
            weights = torch.load(f"model_data/pre_models/{data_type}.pth")  # 75  better_E_final
            self.resnet18.load_state_dict(weights)
        except:
            pass
        self.Init()

    def Init(self,):
        for param in self.resnet18.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet18.conv1(x)
            bn_out_128x128 = self.resnet18.bn1(x)
            bn_out_128x128 = self.resnet18.relu(bn_out_128x128)

            layer1_out_64x64_o = self.resnet18.layer1(x)
            #layer1_out_64x64_o = self.coordconv1(layer1_out_64x64_o)

            layer2_out_32x32_o = self.resnet18.layer2(layer1_out_64x64_o)
            #layer2_out_32x32_o = self.coordconv2(layer2_out_32x32_o)

            layer3_out_16x16_o = self.resnet18.layer3(layer2_out_32x32_o)
            forward_out = self.resnet18.layer4(layer3_out_16x16_o)
        return forward_out,layer1_out_64x64_o,layer2_out_32x32_o,layer3_out_16x16_o,bn_out_128x128


from gcn_lib import Grapher, act_layer
class ProjectionNet(nn.Module): # SLSG model
    def __init__(self,out_features=False,num_classes = 2,data_type=None,use_se=None,use_patch_core = False,encoder_freeze=False,epoch_MODE=0):
        super(ProjectionNet, self).__init__()
        self.epoch_MODE = epoch_MODE
        self.encoder_segment = Encoder(data_type=data_type,use_patch_core = use_patch_core,encoder_freeze=encoder_freeze,epoch_MODE=epoch_MODE)
        self.decoder_segment = Decoder (base_width=64)

        self.aux0 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),)

        self.aux0_out = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 2, kernel_size=3, padding=1),
        nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True))

        self.LKA64 = SpatialAttention(64,3,3)#64,128,128  25
        self.LKA128 = SpatialAttention(128,3,3)#128,64,64  19
        self.LKA256 = SpatialAttention(256,3,3)#256,32,32  13

        mo_list = [self.aux0, self.aux0_out, self.LKA64, self.LKA128, self.LKA256]
        for m in mo_list:
            # for m in self.block_down1.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        in_channels = 512

        from gcn_lib.pos_embed import get_2d_sincos_pos_embed
        self.pos_emd1 = torch.from_numpy(np.float32(get_2d_sincos_pos_embed(64, 128)).reshape(128,128,64).transpose(2,0,1)).cuda()
        self.pos_emd2 = torch.from_numpy(np.float32(get_2d_sincos_pos_embed(128, 64)).reshape(64,64,128).transpose(2,0,1)).cuda()
        self.pos_emd3 = torch.from_numpy(np.float32(get_2d_sincos_pos_embed(256, 32)).reshape(32,32,256).transpose(2,0,1)).cuda()

        self.GCN3 =Grapher(in_channels, 9, 3, 'mr', 'gelu', 'batch',
                    True, False, 0.2, 1, n=16*16, drop_path=0,
                    relative_pos=True,epoch_MODE=epoch_MODE)
        self.drop = torch.nn.Dropout2d(0.05)


    def forward(self, x):
        forward_out, layer1_out_64x64_o, layer2_out_32x32_o, layer3_out_16x16_o,bn_out_128x128 = self.encoder_segment(x)
        layer1_out_64x64_o,layer2_out_32x32_o,layer3_out_16x16_o = self.LKA64(layer1_out_64x64_o),self.LKA128(layer2_out_32x32_o),self.LKA256(layer3_out_16x16_o)

        forward_out = self.GCN3(forward_out)

        AUX_out0 = self.aux0(forward_out)
        AUX_out = self.aux0_out(AUX_out0)

        output_segment,_ = self.decoder_segment(forward_out, layer1_out_64x64_o, layer2_out_32x32_o, layer3_out_16x16_o,bn_out_128x128,x,AUX_out0)
        output_segment = output_segment+AUX_out

        return output_segment,AUX_out,1



if __name__ == '__main__':
    SLSG_model = ProjectionNet(data_type='juice_bottle')
    x = torch.randn(1,3,256,256)
    y = SLSG_model(x)



