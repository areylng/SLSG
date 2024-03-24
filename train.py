#coding=gbk
from pathlib import Path
from tqdm import tqdm
import datetime
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from loss_fun import FocalLoss
from dataset import MVTecAT, Repeat
from cutpaste_normal import  CutPastePlus2
from utils import cut_paste_collate_fn,str2bool
from model import ProjectionNet
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from eval import eval_model


def train_model(data_type="screw", model_dir="models", epochs=7001, pretrained=True, test_epochs=10, freeze_resnet=20, learninig_rate=0.03,
                 optim_name="SGD", batch_size=64, head_layer=8, cutpate_type=CutPastePlus2, device="cuda", workers=8, size=256, args=None,
                 se=False, variant='xxx', use_img_level_loss=False, use_patch_core=False, encoder_freeze=False, CUTPASTE=0, epoch_MODE=0):

    torch.multiprocessing.freeze_support()
    weight_decay = 0.00003
    momentum = 0.9
    model_name = f"model_in_self-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now() )
    print(model_name)
    #augmentation:
    min_scale = 1

    all_train_image_names = sorted(list(( Path(f"C:\datasets/DATA_SETS/dataset_anomaly_detection/") / data_type / "train" / "good").glob("*.png")))
    #sorted((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
    print("loading images")
    train_transform_load = transforms.Compose([])
    train_transform_load.transforms.append(transforms.Resize((int(size * (1/min_scale)), int(size * (1/min_scale))), Image.ANTIALIAS))
    all_train_imgs = Parallel(n_jobs=1)(
        delayed(lambda file: train_transform_load(Image.open(file).convert("RGB")))(file) for file in all_train_image_names)
    print(f"loaded {len(all_train_image_names)} images")

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, saturation=0.1))
    train_transform.transforms.append(cutpate_type(transform = None,args = args,data_type = data_type,
                                                   model_name = model_name,CUTPASTE=CUTPASTE,all_train_imgs=all_train_imgs,
                                                   img_size_O=size,epoch_MODE=epoch_MODE))

    train_data = MVTecAT(f"C:/datasets/DATA_SETS/dataset_anomaly_detection/", data_type, transform = train_transform,
                         size=int(size * (1/min_scale)),model_name = model_name,all_train_image_names=all_train_image_names,all_train_imgs=all_train_imgs)
    dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn,
                             pin_memory=True,)
    writer = SummaryWriter(Path("logdirs") / model_name)
    num_classes = 2
    model = ProjectionNet(num_classes=num_classes,data_type=data_type,use_se=se,use_patch_core = use_patch_core,encoder_freeze=encoder_freeze,epoch_MODE=epoch_MODE)
    model.to(device)

    loss_focal = FocalLoss()
    loss_l1 = torch.nn.L1Loss()
    if optim_name == "sgd":
        optimizer = optim.SGD( model.parameters(), lr=learninig_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learninig_rate, weight_decay=weight_decay)
        scheduler = None

    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out
    dataloader_inf =  get_data_inf()

    if not os.path.isdir(f"./imag_result/{model_name}"):
        os.mkdir(f"./imag_result/{model_name}")
    if not os.path.isdir(f"./imag_result/{model_name}/train"):
        os.mkdir(f"./imag_result/{model_name}/train")

    image_MAX_his_auc = 0
    model.train()
    for step in tqdm(range(epochs)):
        epoch = int(step / 1)
        if epoch == freeze_resnet:
            model.unfreeze()
        batch_idx, data = next(dataloader_inf)
        xs = [x.to(device) for x in data[0:num_classes]]
        if num_classes == 2:
            seg_target = data[2]
            part1 = np.zeros_like(seg_target)
            part1 = torch.Tensor(part1)
            seg_target = torch.cat([part1,seg_target], axis=0)
            seg_target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        xc = torch.cat(xs, axis=0)#batch_size为16 这里的xc维度为48，3，256，256
        output_segment, output_aux, _ = model(xc)

        out_mask_sm = torch.softmax(output_segment, dim=1)
        aux_mask_sm = torch.softmax(output_aux, dim=1)
        segment_loss = loss_focal(out_mask_sm, seg_target)
        # segment_loss_aux = loss_focal(out_mask_sm, seg_target)

        l1_mask = out_mask_sm[:, 1, :, :]
        l1_mask = torch.unsqueeze(l1_mask, 1)
        l1_loss = loss_l1(l1_mask, seg_target.cuda())

        aux_mask_ = aux_mask_sm[:, 1, :, :]
        aux_mask_ = torch.unsqueeze(aux_mask_, 1)
        loss_aux = loss_l1(aux_mask_, seg_target.cuda())

        loss = 0.4 * segment_loss + 0.6 * l1_loss + 0.3 * loss_aux
        # regulize weights:
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)

        if epoch>0 and epoch % test_epochs == 0:
            model.eval()
            per_pixel_rocauc,auc_score_max_50_mean,auc_score_max_100_mean,img_max,SAVE_model = eval_model(model_name, data_type, device=device,
                                save_plots=False,
                                size=size,
                                show_training_data=False,
                                model=model,
                                step = step,
                                Get_feature = '',
                                image_MAX_his_auc = image_MAX_his_auc
                                )
            if auc_score_max_100_mean>image_MAX_his_auc:
                image_MAX_his_auc=auc_score_max_100_mean
            writer.add_scalar('auc/pixel', per_pixel_rocauc, step)
            writer.add_scalar('auc/mean_50', auc_score_max_50_mean, step)
            writer.add_scalar('auc/mean_100', auc_score_max_100_mean, step)
            model.train()
            ss = str(auc_score_max_100_mean)
            torch.save(model.state_dict(), model_dir / f"{model_name}_{epoch}_{ss[:5]}.tch")
    torch.save(model.state_dict(), model_dir / f"{model_name}_final.tch")
    writer.close()
    del model
