'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-03-10 09:39:18
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-03-18 12:13:21
FilePath: /mru/Knowledge-Distillation/main.py
Description: 
    1. load a well-pretrained student model
    2. train a student model from scratch by knowledge distillation
    3. prune the well-trained student model by package torch_pruning
    4. finetune the pruned model
Reference:
    1. https://github.com/DefangChen/SimKD
    2. https://github.com/VainF/Torch-Pruning
'''

# package
import os
import random
import argparse
import numpy as np
from functools import partial

# torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch_pruning as tp

# other .py
from model import resnet14, ResNet_50
from kd_loss import DistillKL
from utils import log_file, logger
from train import train_student, evaluate, pruning, finetune


# --------------------- parameters ---------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--num_classes', default=10, type=int, choices=[10], help='number of predicted classes')
    parser.add_argument('--batch_size', default=2000, type=int, help='batch size')
    parser.add_argument('--criterion_cls', default='CrossEntropyLoss', choices=['CrossEntropyLoss'], help='classification criterion')
    parser.add_argument('--epoch_size', default=300, type=int, help='epoch size')
    parser.add_argument('--finetune_epoch_size', default=300, type=int, help='epoch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--mode', default="all", choices=['All', 'train', 'prune'])
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--num_workers', default=16, type=int, help='number of data loading threads')
    parser.add_argument('--optimizer', default="Adam", choices=['adam', 'rmsprop', 'sgd'])
    parser.add_argument('--schedular', default="ReduceLROnPlateau", choices=['ReduceLROnPlateau'])
    parser.add_argument('--seed', default=777, type=int, help='manual seed')
    parser.add_argument('--val_freq', default=10, type=int, help='validation frequency')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

    # knowledge distillation
    parser.add_argument('--cls_weight', type=float, default=0.8, help='loss weight')
    parser.add_argument('--criterion_kd', default='vanilla_kd', choices=['vanilla_kd'], help='knowledge distillation criterion')
    parser.add_argument('--kd_T', type=float, default=2.0, help='temperature for KD distillation')
    parser.add_argument('--kd_weight', type=float, default=0.4, help='loss weight')
    parser.add_argument('--student_model', default="resnet14", choices=['resnet14'], help='student model')

    # root
    parser.add_argument('--data_root', default='./data', help='root directory for data')
    parser.add_argument('--fname', default=None, help='log directory name')
    parser.add_argument('--log_root', default='./logs', help='root directory for log')
    parser.add_argument('--log_dir', default=None, help='root for student model (for mode prune)')
    parser.add_argument('--teacher_model_path', default="./resnet-50.pth", help='teacher model path')

    # utils
    parser.add_argument('--cuda', default=True, action='store_false')
    parser.add_argument('--device', default="cuda:0", help='GPU number')
    parser.add_argument('--save_model', default=True, action='store_false')
    
    args = parser.parse_args()
    return args


# --------------------- main ---------------------
if __name__ == '__main__':
    
    # --------------------- Parameter ---------------------
    args = parse_args()

    # --------------------- log ---------------------
    if args.mode in ["train", "all"]:
        log_dir = log_file(args.log_root, args.__dict__, args.fname)
    elif args.mode == "prune":
        log_dir = args.log_dir

    # --------------------- Device ---------------------
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = torch.device(args.device)
        print('device : ', args.device, torch.cuda.get_device_name())
    else:
        device = 'cpu'
        print('device : ', device)
    

    # --------------------- Random seed ---------------------
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


    # --------------------- load train / test dataset --------------------- 
    train_dataset = torchvision.datasets.FashionMNIST(root=args.data_root, train=True, download=True,
                                                transform=transforms.Compose(
                                                    [transforms.RandomHorizontalFlip(),
                                                    # transforms.RandomRotation(degrees=(0, 60)),
                                                    transforms.Grayscale(num_output_channels=3),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (0.5,))])
                                                )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers)

    test_dataset = torchvision.datasets.FashionMNIST(root=args.data_root, train=False, download=True,
                                                    transform=transforms.Compose(
                                                        [transforms.Grayscale(num_output_channels=3),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5,), (0.5,))])
                                                    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    
    # --------------------- model & optimizers & criterion & tensorboard ---------------------
    # load teacher model
    teacher_model = ResNet_50()
    checkpoint = torch.load(args.teacher_model_path)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.to(device)

    # initialize student model
    if args.student_model == "resnet14":
        student_model = resnet14(num_classes=args.num_classes)
    student_model.to(device)

    # optimizers
    if args.optimizer.lower() == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer.lower() == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer.lower() == 'sgd':
        args.optimizer = optim.SGD
    try:
        optimizer = args.optimizer(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    except:
        optimizer = args.optimizer(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # lr scheduler
    if args.schedular.lower() == 'ReduceLROnPlateau'.lower():
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # criterion
    if args.criterion_cls.lower() == 'CrossEntropyLoss'.lower():
        criterion_cls = nn.CrossEntropyLoss()
    if args.criterion_kd.lower() == 'vanilla_kd'.lower():
        criterion_kd = DistillKL(args.kd_T)
    
    # tensorboard
    writer = SummaryWriter(os.path.join(log_dir, "tb_record"))


    # --------------------- training loop ---------------------
    if args.mode in ["train", "all"]:

        # logger
        logger("Training", title=True, log_dir=log_dir)
        
        max_accuracy = 0

        for epoch in range(1, args.epoch_size+1):

            # train
            epoch_loss, avg_loss, epoch_loss_kd, epoch_loss_cls, accuracy = train_student(teacher_model, student_model,
                                                                            args.kd_weight, args.cls_weight,
                                                                            epoch, args.epoch_size, train_loader, optimizer,
                                                                            criterion_kd, criterion_cls, device)

            # lr scheduler
            scheduler.step(epoch_loss)
            
            # logger
            logger('epoch : {:03d}/{} \t Epoch loss:{:.6f} \t Average loss:{:.6f} \t Accuracy: {:.4f} \t KD loss: {:.6f} \t CLS loss: {:.6f}'\
                    .format(epoch, args.epoch_size, epoch_loss, avg_loss, accuracy, epoch_loss_kd, epoch_loss_cls),\
                    title=False, log_dir=log_dir)

            # tb record
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Train/loss', epoch_loss, epoch)
            writer.add_scalar('Train/accuracy', accuracy, epoch)


            # --------------------- validation ---------------------
            if epoch % args.val_freq == 0 or epoch == 1:
                
                # evaluate
                accuracy = evaluate(student_model, test_loader, device)
                
                # logger
                if accuracy > max_accuracy:
                    logger(f"### Accuracy on {len(test_loader.dataset)} test images: {accuracy:.4f} Best accuracy so far: {max_accuracy}  =>  saving model... ###", title=False, log_dir=log_dir)
                else:
                    logger(f"### Accuracy on {len(test_loader.dataset)} test images: {accuracy:.4f} Best accuracy so far: {max_accuracy} ###", title=False, log_dir=log_dir)

                # tb record
                writer.add_scalar('Train/val_accuracy', accuracy, epoch)

                # --------------------- save model ---------------------
                if args.save_model and accuracy > max_accuracy:
                    torch.save(student_model.state_dict(), os.path.join(log_dir, 'student_model.pt'))
                    max_accuracy = accuracy


    # --------------------- pruning ---------------------
    if args.mode in ["prune", "all"]:
        
        # logger
        logger("Pruning", title=True, log_dir=log_dir)
        
        # load best model
        pruned_model = resnet14(num_classes=args.num_classes)
        pruned_model.load_state_dict(torch.load(os.path.join(log_dir, 'student_model.pt')))

        # prune
        pruned_model, ori_size, ori_acc, pruned_size, pruned_acc = pruning(pruned_model, device, train_dataset, train_loader, test_loader, args.num_classes)
        
        # logger
        logger(f"Original Model Size: {ori_size:6d}, Original Accuracy on Testing Img: {ori_acc:.4f}", title=False, log_dir=log_dir)
        logger(f"Pruned Model Size: {pruned_size:6d}, Pruned Accuracy on Testing Img: {pruned_acc:.4f}", title=False, log_dir=log_dir)


        # --------------------- finetuning loop ---------------------
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                                    shuffle=True, num_workers=args.num_workers)

        # logger
        logger("Finetuning", title=True, log_dir=log_dir)

        max_accuracy = 0

        # optimizer, scheduler
        optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 200], gamma=0.1)
        
        for epoch in range(1, args.finetune_epoch_size+1):
            
            # finetune
            epoch_loss, avg_loss, accuracy = finetune(pruned_model, epoch, args.finetune_epoch_size, train_loader, optimizer, criterion_cls, device)

            # lr scheduler
            scheduler.step()
            
            # tb record
            writer.add_scalar('Finetune/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Finetune/loss', epoch_loss, epoch)
            writer.add_scalar('Finetune/accuracy', accuracy, epoch)


            # --------------------- validation ---------------------
            # evaluate
            test_accuracy = evaluate(pruned_model, test_loader, device)
            
            # logger
            logger('epoch : {:03d}/{} \t Epoch loss:{:.6f} \t Average loss:{:.6f} \t Accuracy: {:.4f} \t Testing Accuracy: {:.4f}'\
                    .format(epoch, args.finetune_epoch_size, epoch_loss, avg_loss, accuracy, test_accuracy),\
                    title=False, log_dir=log_dir)

            # tb record
            writer.add_scalar('Finetune/val_accuracy', test_accuracy, epoch)

            # --------------------- save model ---------------------
            if args.save_model and test_accuracy > max_accuracy:
                logger("### saving best model ###", title=False, log_dir=log_dir)
                torch.save(pruned_model, os.path.join(log_dir, 'pruned_model.pth'))
                max_accuracy = test_accuracy
    
    writer.close()

    # -------------- Training finished --------------
    open(os.path.join(log_dir, 'done'), 'a').close()
    logger("Finished training", title=True, log_dir=log_dir)