"""
Reference:
    1. https://github.com/VainF/Torch-Pruning
"""
import os
from tqdm import tqdm
from functools import partial
import torch
import torch.nn.functional as F
import torch_pruning as tp

# --------------------- train student ---------------------
def train_student(model_t, model_s, kd_weight, cls_weight, epoch, total_epoch, train_loader, optimizer, criterion_kd, criterion_cls, device):
    
    # eval
    model_t.eval()

    # train
    model_s.train()

    # params
    epoch_loss, epoch_loss_kd, epoch_loss_cls, epoch_correct = 0, 0, 0, 0

    with tqdm(train_loader, ncols=0, leave=False) as pbar:

        pbar.set_description(f"Epoch {epoch:03d}/{total_epoch:3d}")

        for (images, labels) in pbar:

            # put on device (usually GPU)
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # predict
            pred_s = model_s(images)
            with torch.no_grad():
                pred_t = model_t(images)

            # loss
            loss_cls = criterion_cls(pred_s, labels)
            loss_kd = criterion_kd(pred_s, pred_t)
            loss = cls_weight * loss_cls + kd_weight * loss_kd

            # the label with the highest energy will be our prediction
            _, predicted = torch.max(pred_s.data, 1)
            correct_pred = (predicted == labels).sum().item()

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss & acc
            epoch_loss += loss.detach().cpu().numpy()
            epoch_loss_kd += loss_kd.detach().cpu().numpy()
            epoch_loss_cls += loss_cls.detach().cpu().numpy()
            epoch_correct += correct_pred
    
    accuracy = epoch_correct / len(train_loader.dataset)
    avg_loss = float(epoch_loss) / (len(train_loader.dataset))

    return epoch_loss, avg_loss, epoch_loss_kd, epoch_loss_cls, accuracy


# --------------------- finetuning ---------------------
def finetune(model, epoch, total_epoch, train_loader, optimizer, criterion, device):

    # train
    model.train()

    # params
    epoch_loss, epoch_correct = 0, 0

    with tqdm(train_loader, ncols=0, leave=False) as pbar:

        pbar.set_description(f"Epoch {epoch:03d}/{total_epoch:3d}")

        for (images, labels) in pbar:

            # put on device (usually GPU)
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # predict
            pred = model(images)

            # loss
            loss = F.cross_entropy(pred, labels)

            # the label with the highest energy will be our prediction
            _, predicted = torch.max(pred.data, 1)
            correct_pred = (predicted == labels).sum().item()

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss & acc
            epoch_loss += loss.detach().cpu().numpy()
            epoch_correct += correct_pred
    
    accuracy = epoch_correct / len(train_loader.dataset)
    avg_loss = float(epoch_loss) / (len(train_loader.dataset))

    return epoch_loss, avg_loss, accuracy


# --------------------- evaluate ---------------------
def evaluate(model, test_loader, device):

    # param
    val_correct = 0
    
    # switch to mode evaluation
    model.eval()

    with torch.no_grad():
        for (images, labels) in test_loader:

            # put on device (usually GPU)
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # predict
            pred = model(images)

            # the label with the highest energy will be our prediction
            _, predicted = torch.max(pred.data, 1)
            correct_pred = (predicted == labels).sum().item()

            # accumulating correct term
            val_correct += correct_pred
    
    accuracy = val_correct / len(test_loader.dataset)

    return accuracy


# --------------------- pruning ---------------------
def pruning(model, device, train_dataset, train_loader, test_loader, num_classes):
        
    # GPU
    model = model.to(device)
    
    # evaluation mode
    model.eval()

    # example input (for pruning)
    example_inputs = train_dataset[0][0].unsqueeze(0).to(device)

    # evaluate the original acc, model size, and get the pruner
    _, ori_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    ori_acc = evaluate(model, test_loader, device)
    pruner = get_pruner(model, example_inputs, num_classes)

    # pruninig
    progressive_pruning(pruner, model, speed_up=2, example_inputs=example_inputs)
    del pruner

    # evaluate the acc, model size after pruning
    _, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    pruned_acc = evaluate(model, test_loader, device=device)
    
    return model, ori_size, ori_acc, pruned_size, pruned_acc


def get_pruner(model, example_inputs, num_classes):

    imp = tp.importance.GroupNormImportance(p=2)
    pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=False)

    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    ch_sparsity_dict = {}

    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == num_classes:
            ignored_layers.append(m)
    
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=400,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=1.0,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner


def progressive_pruning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
    return current_speed_up