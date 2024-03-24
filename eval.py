#coding=gbk
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MVTecAT, Repeat
from sklearn.metrics import roc_auc_score
import numpy as np
from PIL import Image
from mvtec_loco_evaluation.evaluate_experiment import EVAL_ME
import cv2

test_data_eval = None
test_transform = None
cached_type = None

def eval_model(modelname, data_type, device="cpu", save_plots=False, size=512, show_training_data=True, model=None,
               train_embed=None, head_layer=8, step=0   , Get_feature = "Get_feature",image_MAX_his_auc=0
   ):
    model.eval()
    global test_data_eval, test_transform, cached_type
    if test_data_eval is None or cached_type != data_type:
        cached_type = data_type
        test_transform = transforms.Compose([])
        test_transform.transforms.append(transforms.Resize((size,size), Image.ANTIALIAS))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]))
        test_data_eval = MVTecAT(f"C:\datasets/DATA_SETS/dataset_anomaly_detection/", data_type, size, transform=test_transform, mode="test")

    dataloader_test = DataLoader(test_data_eval, batch_size=16,
                                 shuffle=False, num_workers=0)


    # get embeddings for test data
    labels = []
    output_segments = []
    logits = []
    true_masks = []
    with torch.no_grad():
        index_ = 0
        for x, label, img_mask in dataloader_test:  # x维度为B,3,256,256
            x = x.to(device)
            output_segment, logit, layer_final_256x256 = model(x)
            output_segment = (output_segment)
            true_masks.append(img_mask.cpu())
            # save
            output_segments.append(output_segment.cpu())
            labels.append(label.cpu())
    labels = torch.cat(labels)  # 83
    output_segments = torch.cat(output_segments)  # 83,512
    output_segments = torch.softmax(output_segments, dim=1)


    true_masks = torch.cat(true_masks)  # 83,512
    true_masks = true_masks.numpy()
    output_segments = output_segments.numpy()
    output_segments = output_segments[:, 1, :, :]
    import time
    # s = time.time()
    # spro_auc = Get_spro_score(output_segments,data_type,MAXstep=100)
    # e = time.time()
    # print('Run',(e-s))

    #Get AUC from seg:
    MAX_anormaly = []
    for im_index in range(output_segments.shape[0]):
         MAX_anormaly.append(output_segments[im_index].max())

    all_auc = []
    auc_score_max = roc_auc_score(labels, np.array(MAX_anormaly))
    all_auc.append(auc_score_max)
    MAX_anormaly_100 = []
    if not os.path.isdir(f"./imag_result/{modelname}"):
        os.mkdir(f"./imag_result/{modelname}")
    if not os.path.isdir(f"./imag_result/{modelname}/test"):
        os.mkdir(f"./imag_result/{modelname}/test")
    if not os.path.isdir(f"./imag_result/{modelname}/test/{step}"):
        os.mkdir(f"./imag_result/{modelname}/test/{step}")


    for im_index in range(output_segments.shape[0]):
         tempp = output_segments[im_index].flatten()
         tempp.sort()
         pixel_num = size*size
         MAX_anormaly_100.append(tempp[pixel_num-100:pixel_num].mean())
    auc_score_max_100_mean = roc_auc_score(labels, np.array(MAX_anormaly_100))
    all_auc.append(auc_score_max_100_mean)

    img_max = np.array(all_auc).max()
    if data_type in ['juice_bottle', 'breakfast_box', 'splicing_connectors', 'pushpins', 'screw_bag']:
        if data_type=='juice_bottle':
            spilt = [94,94+142]
            img_size = [800,1600]
            img_max_com = 0.925
        if data_type == 'splicing_connectors':
            spilt = [119, 119 + 108]
            img_size = [1700, 850]
            img_max_com = 0.79
        if data_type == 'screw_bag':
            spilt = [122, 122 + 137]
            img_size = [1600, 1100]
            img_max_com = 0.74
        if data_type == 'pushpins':
            spilt = [138, 138 + 91]
            img_size = [1700, 1000]
            img_max_com = 0.91
        if data_type == 'breakfast_box':
            spilt = [102, 102 + 83]
            img_size = [1600, 1280]
            img_max_com = 0.86

        MAX_MAX = MAX_anormaly_100#All_max[img_max_index]
        LOCO_pre = MAX_MAX[0:spilt[1]]
        LOCO_True = labels[0:spilt[1]]
        STRU_Pre = MAX_MAX[0:spilt[0]]+MAX_MAX[spilt[1]:]
        STRU_True = list(labels[0:spilt[0]])+list(labels[spilt[1]:])
        LOCO_AUC_IMG = roc_auc_score(LOCO_True, np.array(LOCO_pre))
        STRU_AUC_IMG = roc_auc_score(np.array(STRU_True), np.array(STRU_Pre))

        eval_pixel = False
        if eval_pixel == True:
            if (img_max > image_MAX_his_auc and img_max>=img_max_com) or step%2000==0:
                    output_segments_NEW = output_segments
                    all_index_img = 0
                    index_img = 0
                    for ikik in range(0,spilt[0]):
                        temp = output_segments_NEW[ikik]
                        temp = cv2.resize(temp, (img_size[0], img_size[1]))
                        tt = str(index_img)
                        if len(tt)==1:
                            tt='00'+tt
                        if len(tt)==2:
                            tt='0'+tt
                        cv2.imwrite(f'log_metris/model_name/{data_type}/test/good/{tt}.tiff', temp)
                        index_img+=1
                        all_index_img+=1

                    index_img = 0
                    for ikik in range(spilt[0],spilt[1]):
                        temp = output_segments_NEW[ikik]
                        temp = cv2.resize(temp,(img_size[0], img_size[1]))
                        tt = str(index_img)
                        if len(tt)==1:
                            tt='00'+tt
                        if len(tt)==2:
                            tt='0'+tt
                        cv2.imwrite(f'log_metris/model_name/{data_type}/test/logical_anomalies/{tt}.tiff', temp)
                        index_img+=1
                        all_index_img += 1
                    index_img = 0
                    for ikik in range(spilt[1], output_segments.shape[0]):
                        temp = output_segments_NEW[ikik]
                        temp = cv2.resize(temp, (img_size[0], img_size[1]))
                        tt = str(index_img)
                        if len(tt)==1:
                            tt='00'+tt
                        if len(tt)==2:
                            tt='0'+tt
                        cv2.imwrite(f'log_metris/model_name/{data_type}/test/structural_anomalies/{tt}.tiff', temp)
                        index_img+=1
                        all_index_img += 1
                    print('generate done')
                    EVAL_ME(data_type=data_type)
    pro_auc=-1
    pro_auc_30 = -1
    SAVE_model = False

    vis = False
    if vis:
        for iiii in range(output_segments.shape[0]):
            aaaty = (output_segments[iiii, :, :]*255)
            cv2.imwrite(f"./imag_result/{modelname}/test/{step}/{step}_{iiii}.jpg", aaaty,)

    true_masks = true_masks.flatten().astype(np.uint32)
    output_segments = output_segments.flatten()

    per_pixel_rocauc = roc_auc_score(true_masks, output_segments)
    print(data_type)
    if data_type in ['juice_bottle', 'breakfast_box', 'splicing_connectors', 'pushpins', 'screw_bag']:
        print("---------------------------------------Test_AUC_Img ",auc_score_max_100_mean,'L&S', LOCO_AUC_IMG ,STRU_AUC_IMG)
    else:
        print("---------------------------------------Test_AUC_Img ", auc_score_max_100_mean, '               ', auc_score_max)
    print("---------------------------------------Test_AUC_Pix ",
          per_pixel_rocauc)
    if pro_auc>=0:
        print("-------------------------------------------------------------------------------Test_pixel_PRO", pro_auc)
        print("-------------------------------------------------------------------------------Test_pixel_30%", pro_auc_30)
    auc_score_max_50_mean = 0
    return  per_pixel_rocauc,auc_score_max_50_mean,auc_score_max_100_mean,img_max,SAVE_model
