import argparse
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import sys
import os
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)


from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, validate
from mdistiller.engine.cfg import CFG as cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="resnet32x4")
    parser.add_argument("-m1", "--model1", type=str, default="resnet8x4")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain") # "pretrain"
    parser.add_argument("-c1", "--ckpt1", type=str, default="path/to/trained_ckpt") # "pretrain"
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet", "tiny_imagenet"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    if args.dataset == "imagenet":
        val_loader = get_imagenet_val_loader(args.batch_size)
        if args.ckpt == "pretrain":
            model = imagenet_model_dict[args.model](pretrained=True)
        else:
            model = imagenet_model_dict[args.model](pretrained=False)
            model.load_state_dict(load_checkpoint(args.ckpt)["model"])
    elif args.dataset in ("cifar100", "tiny_imagenet"):
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model_dict = cifar_model_dict
        # teacher
        model, pretrain_model_path = model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])
        # student
        model_student, pretrain_model_path_student = model_dict[args.model1]
        model_student = model_student(num_classes=num_classes)
        ckpt1 = pretrain_model_path if args.ckpt1 == "pretrain" else args.ckpt1
        model_student.load_state_dict(load_checkpoint(ckpt1)["model"])

    # teacher
    model = Vanilla(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    # student
    model_student = Vanilla(model_student)
    model_student = model_student.cuda()
    model_student = torch.nn.DataParallel(model_student)

    # test_acc, test_acc_top5, test_loss = validate(val_loader, model)
    distiller = model
    if hasattr(val_loader, 'loader'):
        num_iter = len(val_loader.loader)
    else:
        num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    temp = 0
    with torch.no_grad():
        for idx, (image,target) in enumerate(val_loader):
            # image, target = data[0]["data"], data[0]["label"].squeeze(-1).long()
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # with torch.cuda.amp.autocast():
            # print(image.shape, target.shape)
            output = distiller(image=image)
            softmax_output = torch.nn.functional.softmax(output, dim=1)
            sorted_softmax_output, _ = torch.sort(softmax_output, dim=1, descending=True)
            output_mean = sorted_softmax_output.mean(dim=0)
            temp = temp + output_mean
    import numpy as np
    np.savez("res32x4_output.npz", output=(temp / len(val_loader)).cpu().data.numpy())
        # print(temp / len(val_loader))


