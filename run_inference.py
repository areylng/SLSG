#coding=gbk
from pathlib import Path
import datetime
import argparse
import torch
from cutpaste_normal import CutPastePlus2
from model import ProjectionNet
from utils import str2bool
from eval import eval_model


def run_infer(data_type="screw", model_dir="models", epochs=256, pretrained=True, test_epochs=10, freeze_resnet=20, learninig_rate=0.03,
                 optim_name="SGD", batch_size=64, head_layer=8, cutpate_type=CutPastePlus2, device="cuda", workers=8, size=256, args=None,
                 se=False, variant='xxx', use_img_level_loss=False, use_patch_core=False, encoder_freeze=False, CUTPASTE=0, epoch_MODE=0):

    torch.multiprocessing.freeze_support()
    model_name = f"model_in_self-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
    num_classes = 2
    model = ProjectionNet(num_classes=num_classes, data_type=data_type, use_se=se, use_patch_core=use_patch_core, encoder_freeze=encoder_freeze, epoch_MODE=epoch_MODE)
    weights = torch.load(f"model_data/{data_type}.tch")
    model.load_state_dict(weights)
    model.to(device)
    step = 0
    image_MAX_his_auc = 0
    model.eval()
    _, _, _, _, _ = eval_model(model_name,data_type,device=device,save_plots=False,size=size,show_training_data=False,
                              model=model,step=step,Get_feature='',image_MAX_his_auc=image_MAX_his_auc)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="all", help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')
    parser.add_argument('--epochs', default=3701, type=int, help='number of epochs to train the model , (default: 256)')
    parser.add_argument('--model_dir', default="models", help='output folder of the models , (default: models)')
    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false', help='use pretrained values to initalize ResNet18 , (default: True)')
    parser.add_argument('--test_epochs', default=100, type=int, help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')
    parser.add_argument('--freeze_resnet', default=60000, type=int, help='number of epochs to freeze resnet (default: 20)')
    parser.add_argument('--lr', default=0.5, type=float, help='learning rate (default: 0.03)')
    parser.add_argument('--optim', default="sgd", help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')
    parser.add_argument('--head_layer', default=1, type=int,help='number of layers in the projection head (default: 1)')
    parser.add_argument('--variant', default="plus2", choices=['normal', 'scar', '3way', 'union', 'plus', 'plus2'],help='cutpaste variant to use (dafault: "3way")')
    parser.add_argument('--cuda', default=True, type=str2bool, help='use cuda for training (default: False)')
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


    variant_map = {"plus2": CutPastePlus2}
    variant = variant_map[args.variant]

    device = "cuda" if args.cuda else "cpu"
    print(f"using device: {device}")

    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))

    for epoch_MODE in [0]:
        for data_type in types:
            args.epochs = 1
            encoder_freeze = True
            print(f"======================================={data_type}_{epoch_MODE}=======================================")
            torch.cuda.empty_cache()
            run_infer(data_type,
                         model_dir=Path(args.model_dir),
                         epochs=args.epochs,
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
                         variant=args.variant,
                         args=args,
                         se=False,
                         use_img_level_loss=False,
                         use_patch_core=False,
                         encoder_freeze=encoder_freeze,
                         CUTPASTE=99,
                         epoch_MODE=epoch_MODE,
                         )

