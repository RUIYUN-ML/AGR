import torch 
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import attack 
import time 
import model
from torchtoolbox.transform import Cutout
import AGR
import argparse
from tools import *
from awp import AdvWeightPerturb
from mart import mart_loss
from tiny_imagenet_load import TinyImageNet
import numpy as np
import os
parser = argparse.ArgumentParser(description='Hyperparameters')

#Hyperparameter setting 
parser.add_argument('--lr', default=0.1, help='Learning rate', type=float)
parser.add_argument('--momentum', default=0.9, help='Momentum', type=float)
parser.add_argument('--weight_decay', default=5e-4, help='Weight_decay', type=float)
parser.add_argument('--batch_size', default=128, help='Batch size', type=int)
parser.add_argument('--epochs', default=200, help='Number of epochs', type=int)
parser.add_argument('--optimizer', default='sgd', help='Optimizer', choices=['sgd', 'adam'], type=str)
parser.add_argument('--device', default='cuda:0', help='Selection of cuda', choices=['cuda:0','cuda:1'], type=str)
parser.add_argument('--num_classes', default=10, help='Number of classes', type=int)
parser.add_argument('--num_RC', default=32, help='Number of RandomCrop',choices=[32,28], type=int)
parser.add_argument('--num_pgd_eval', default=20, help='Number of PGD in eval', type=int)
parser.add_argument('--lr_stage', default=[100,150], help='Stage setting of learing rate')
parser.add_argument('--model', default='preactresnet', help='Selection of model',choices=['preactresnet','wideresnet'], type=str)
parser.add_argument('--dataset', default='CIFAR10', help='Selection of dataset',choices=['CIFAR10','CIFAR100','tiny-imagenet'], type=str)
parser.add_argument('--AGR', help='Whether to use AGR', action='store_true')
parser.add_argument('--method', default='AT', help='Selection of methods of adversarial training',choices=['trades','awp','AT','mart','Avmixup'], type=str)
parser.add_argument('--gamma', default=0.01, help='Hyperparameter of the awp', type=float)
parser.add_argument('--lr_proxy', default=0.01, help='Learning rate of proxy optim', type=float)
parser.add_argument('--clip_norm', default=0.1, help='Threshold of clipping', type=float)

args = parser.parse_args()
device =torch.device(args.device if torch.cuda.is_available() else "cpu")
print('The current use is:{}'.format(device))

if args.dataset == 'CIFAR100':
    args.num_classes = 100
elif args.dataset == 'tiny-imagenet':
    args.num_classes = 200
    
if args.dataset == 'CIFAR10':
    train_dataset =torchvision.datasets.CIFAR10(root="./data",train=True,transform=transforms.Compose([transforms.Resize(32),transforms.RandomCrop(args.num_RC,padding=4),transforms.RandomHorizontalFlip(),Cutout(),
                                                                                                   transforms.ToTensor()]),
                                            download=True)
    test_dataset =torchvision.datasets.CIFAR10(root="./data",train=False,transform=transforms.Compose([transforms.Resize(32),
                                                                                                   transforms.ToTensor()]),
                                           download=True)

elif args.dataset == 'CIFAR100':
    train_dataset =torchvision.datasets.CIFAR100(root="./data",train=True,transform=transforms.Compose([transforms.Resize(32),transforms.RandomCrop(args.num_RC,padding=4),transforms.RandomHorizontalFlip(),Cutout(),
                                                                                                   transforms.ToTensor()]),
                                            download=True)
    test_dataset =torchvision.datasets.CIFAR100(root="./data",train=False,transform=transforms.Compose([transforms.Resize(32),
                                                                                                   transforms.ToTensor()]),
                                           download=True)

elif args.dataset == 'tiny-imagenet':
    train_dataset = TinyImageNet('dataset_dir', train=True,transform=transforms.Compose([transforms.Resize(32),transforms.RandomCrop(args.num_RC,padding=4),transforms.RandomHorizontalFlip(),Cutout(),
                                                                                                   transforms.ToTensor()]))
    test_dataset = TinyImageNet('dataset_dir', train=False,transform=transforms.Compose([transforms.Resize(32),
                                                                                                   transforms.ToTensor()]))
train_len = len(train_dataset)
test_len = len(test_dataset)

#loading the dataset
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=False,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,drop_last=False,shuffle=False)

agr = AGR.AGR()

if args.model == 'preactresnet':
    Model = model.PreActResNet18(args.num_classes)
    Model.to(device)
    if args.method == 'awp':
        proxy = model.PreActResNet18(args.num_classes)
        proxy.to(device)

elif args.model == 'wideresnet':
    Model = model.wide_resnet_34_10(args.num_classes)
    Model.to(device)
    if args.method == 'awp':
        proxy = model.wide_resnet_34_10(args.num_classes)
        proxy.to(device)
        
PGD = attack.PGD(devices=device)
PGD_eval = attack.PGD(devices=device,step=args.num_pgd_eval)
loss_fc = nn.CrossEntropyLoss()
loss_fc.to(device)

optim = torch.optim.SGD(Model.parameters(),lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)

if args.method == 'awp':
    proxy_opt = torch.optim.SGD(proxy.parameters(),lr=args.lr_proxy)
    awp_adv = AdvWeightPerturb(model=Model,proxy=proxy,proxy_optim=proxy_opt,gamma=args.gamma)
   
if args.AGR:
    result_file_name = "./result/{}_AGR/{}_{}_{}_adv_lr_={}_epochs={}_AGR.txt".format(args.method,args.model,args.dataset,args.method,args.lr,args.epochs)
else:   
    result_file_name = "./result/{}/{}_{}_{}_adv_lr_={}_epochs={}.txt".format(args.method,args.model,args.dataset,args.method,args.lr,args.epochs)
os.makedirs(os.path.dirname(result_file_name), exist_ok=True) 

clip_norm = args.clip_norm
for i in range(args.epochs):
    starttime = time.time()
    print("---------Round {} training---------".format(i+1))
    if i in args.lr_stage:
        optim.param_groups[0]["lr"] *= 0.1 
        
    train_acc_all = 0
    test_acc_all = 0
    test_clean_acc_all = 0
    sim_all = 0
    dis_all = 0
    gop_rate_all = 0
    
    round = 0
    norm_g1_all = 0
    norm_g2_all = 0
    
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)       
        targets = targets.to(device)
        
        imgs_adv = PGD(Model,imgs,targets)
           
        Model.train()
        
        outputs = Model(imgs)
        
        if args.method == 'mart':
            loss_adv, loss_mart = mart_loss(Model, imgs, imgs_adv, targets, optim, device)
            
        loss_train_clean = loss_fc(outputs,targets)
        
        if args.method == 'awp':
            awp_p = awp_adv.calc_awp(inputs_adv=imgs_adv,targets=targets)
            awp_adv.perturb(awp_p)
        
        output_adv = Model(imgs_adv)
        
        if args.method == 'AT':
            loss_train_adv = loss_fc(output_adv,targets)
        elif args.method == 'trades':
            loss_train_adv = 6*kl_div(output_adv,outputs) + loss_fc(outputs,targets)
        elif args.method == 'awp':
            loss_train_adv = loss_fc(output_adv,targets)    
        elif args.method == 'mart':
            loss_train_adv = loss_mart + loss_adv
        elif args.method =='Avmixup':
            onehot = torch.eye(args.num_classes,device=device)[targets].to(device)
            pert =  ( imgs_adv - imgs )*2.0
            x_av = imgs + pert
            x_av = x_av.clamp(0., 1.)
            y_nat = label_smoothing(onehot, args.num_classes, 0.5)
            y_ver = label_smoothing(onehot, args.num_classes, 0.7)
            policy_x = torch.from_numpy(np.random.beta(1, 1, [imgs.size(0), 1, 1, 1])).float().to(device)
            policy_y = policy_x.view(imgs.size(0), -1).to(device)
            x = policy_x * imgs + (1 - policy_x) * x_av
            y = policy_y * y_nat + (1 - policy_y) * y_ver
            out = Model(x)
            loss_train_adv = loss_fc(out,targets)
        
        """
        Conduct the AGR algorithm
        """    
        
        if args.AGR :       
            clean_g = agr.get_gradient(optim,loss_train_clean,Model)
            ori_adv_gradient = agr.get_gradient(optim,loss_train_adv,Model)
            cos_sim_all = 0

            l2_dis_all = 0
            norm_g1_ = 0
            norm_g2_ = 0
            r = 0
            k = 0
            pointer = 0  
            for param in Model.parameters():     
                num_param = param.numel()
                g1 = ori_adv_gradient[pointer:pointer + num_param]
                g2 = clean_g[pointer:pointer + num_param]
                norm_g2 = torch.norm(g2)
                norm_g1 = torch.norm(g1)
                norm_g1_ += norm_g1
                norm_g2_ += norm_g2 
                cs = agr.get_cos_similarity(g1,g2)
                l2_dis_all += agr.get_l2_distance(g1,g2)
                cos_sim_all += cs
                # result_cos.write('{:.3f}, '.format(cs))
                if cs <= 0 :
                    dot_product = torch.dot(g1, g2)
                    PV = (dot_product/(norm_g2**2)) * g2        
                    if i >= 155 :
                        clip_grad_norm = clip_norm/torch.norm(g1-PV)
                        grad = (g1-PV) * torch.min(torch.ones_like(clip_grad_norm).to(device),clip_grad_norm)
                    else :
                        grad = g1
                    param.grad = (grad).view_as(param).clone()
                    k += 1
                    
                else:
                    if i >= 155:
                        grad = (cs)*g2 + (1-cs)*g1
                        clip_grad_norm = clip_norm/torch.norm(grad)
                        grad = grad * torch.min(torch.ones_like(clip_grad_norm).to(device),clip_grad_norm)
                    else : 
                        grad = g1
                    param.grad = (grad).view_as(param).clone()
                pointer += num_param
                r += 1
                    
            gop_rate_all += k/r
            sim_all += cos_sim_all/r
            dis_all += l2_dis_all/r

            norm_g1_all += norm_g1_/r
            norm_g2_all += norm_g2_/r
            round = round + 1
        else:
            optim.zero_grad()
            loss_train_adv.backward()
        optim.step()
        
        if args.method == 'awp':
            awp_adv.restore(awp_p)
            
        train_acc = (output_adv.argmax(1) == targets ).sum()
        train_acc_all = train_acc_all + train_acc.item()
        
    Model.eval()
    for data in test_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        outputs = Model(imgs)
        imgs_eval_adv = PGD_eval(Model,imgs,targets)
        output_eval_adv = Model(imgs_eval_adv)
        
        loss_test_clean = loss_fc(outputs,targets)
        loss_test_adv = loss_fc(output_eval_adv,targets)
        
        test_acc = (output_eval_adv.argmax(1) == targets ).sum()
        test_acc_all = test_acc_all + test_acc.item()
        
        test_clean_acc = (outputs.argmax(1) == targets ).sum()
        test_clean_acc_all = test_clean_acc_all + test_clean_acc.item()
        
    endtime = time.time()        
    
    result_file = open(result_file_name,'a')
    
    if args.AGR:
        result_file.write("{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(i+1,train_acc_all/train_len,test_acc_all/test_len,loss_train_clean
                                                                                    ,loss_train_adv,loss_test_clean,loss_test_adv,test_clean_acc_all/test_len,sim_all/round,gop_rate_all/round,
                                                                                    norm_g1_all/round,norm_g2_all/round,norm_g1_all/round - norm_g2_all/round))
    else:result_file.write("{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(i+1,train_acc_all/train_len,test_acc_all/test_len,loss_train_clean
                                                                                    ,loss_train_adv,loss_test_clean,loss_test_adv,test_clean_acc_all/test_len))
    result_file.close()
    
    print("----------------------------------------------------")
    if args.AGR:
        print("epoch:{}\nmethod:{}\nlr:{}\ntrain_acc:{}\ntest_acc:{}\ntest_clean_acc:{}\nsimilarity:{:.4f}\nl2_dis:{:.4f}\ngop_rate:{:.4f}\nadv_norm:{:.4f}\nclean_norm:{:.4f}\nnorm_diff:{:.4f}\ntime:{:.2f}".format(i+1,args.method,optim.param_groups[0]["lr"],
                                                                                                             train_acc_all/train_len,test_acc_all/test_len,
                                                                                                             test_clean_acc_all/test_len,sim_all/round,
                                                                                                             dis_all/round,gop_rate_all/round,
                                                                                                             norm_g1_all/round,norm_g2_all/round,norm_g1_all/round - norm_g2_all/round,
                                                                                                             endtime-starttime))
    else:
        print("epoch:{}\nmehtod:{}\nlr:{}\ntrain_acc:{}\ntest_acc:{}\ntest_clean_acc:{}\ntime:{:.2f}".format(i+1,args.method,optim.param_groups[0]["lr"],train_acc_all/train_len,test_acc_all/test_len,test_clean_acc_all/test_len,endtime-starttime))
    print("----------------------------------------------------")


if args.AGR :
    save_model_path = "./saved_model/{}_AGR/{}_{}_{}_lr={}_epochs={}_AGR.pt".format(args.method,args.model,args.dataset,args.method,args.lr,args.epochs)
else:
    save_model_path = "./saved_model/{}/{}_{}_{}_lr={}_epochs={}.pt".format(args.method,args.model,args.dataset,args.method,args.lr,args.epochs)
os.makedirs(os.path.dirname(save_model_path), exist_ok=True) 

torch.save(Model.state_dict(),save_model_path)


