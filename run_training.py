#coding=utf-8
import torch
from cutpaste_normal import CutPastePlus2
from utils import str2bool
from train import train_model
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="all",help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')
    parser.add_argument('--epochs', default=3501, type=int,help='number of epochs to train the model , (default: 256)')
    parser.add_argument('--model_dir', default="logdirs",help='output folder of the models , (default: models)')
    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',help='use pretrained values to initalize ResNet18 , (default: True)')
    parser.add_argument('--test_epochs', default=100, type=int,help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')
    parser.add_argument('--freeze_resnet', default=60000, type=int,help='number of epochs to freeze resnet (default: 20)')
    parser.add_argument('--lr', default=0.04, type=float,help='learning rate (default: 0.03)')
    parser.add_argument('--optim', default="sgd",help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')
    parser.add_argument('--batch_size', default=4, type=int,help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')
    parser.add_argument('--head_layer', default=1, type=int, help='number of layers in the projection head (default: 1)')
    parser.add_argument('--variant', default="plus2", choices=['normal', 'scar', '3way', 'union' ,'plus','plus2'], help='cutpaste variant to use (dafault: "3way")')
    parser.add_argument('--cuda', default=True, type=str2bool,help='use cuda for training (default: False)')
    parser.add_argument('--workers', default=0, type=int, help="number of workers to use for data loading (default:8)")

    args = parser.parse_args()
    print(args)
    types = [
        'breakfast_box',
        'juice_bottle',
        'splicing_connectors',
        'pushpins',
        'screw_bag',
    ]
    
    variant_map = {"plus2":CutPastePlus2}
    variant = variant_map[args.variant]
    
    device = "cuda" if args.cuda else "cpu"
    print(f"using device: {device}")
    
    # create modle dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))

    for epoch_MODE in [0]:
        for data_type in types:
                args.lr = 0.04
                # X

                print(f"======================================={data_type}_{epoch_MODE}=======================================")
                torch.cuda.empty_cache()
                train_model(data_type,
                             model_dir=Path(args.model_dir),
                             pretrained=args.pretrained,
                             test_epochs=args.test_epochs,
                             freeze_resnet=args.freeze_resnet,
                             learninig_rate=args.lr,
                             optim_name=args.optim,
                             batch_size=args.batch_size,
                             head_layer=args.head_layer,
                             device=device,
                             cutpate_type=variant,
                             workers=args.workers,
                             variant = args.variant,
                             args = args,
                             se=False,
                             use_img_level_loss = False,
                             use_patch_core = False,
                             encoder_freeze = True,
                             CUTPASTE = 99,
                             epoch_MODE=epoch_MODE,)


