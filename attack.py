import torch
import torch.nn as nn
from torch.autograd import grad
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
class PGD():
    def __init__(self,devices,eps=8/255,alpha=2/255,step = 10, random_start=True):
        self.eps = eps 
        self.alpha = alpha
        self.device = devices 
        self.step = step
        self.random_start = random_start

    def forward(self,model,x,y):
        model.eval()
        image = x.clone().to(self.device).detach()
        target = y.clone().to(self.device).detach()
        loss_fn = nn.CrossEntropyLoss()
        'start at a uniform point'
        image_adv = x.clone().to(self.device).detach()
        if self.random_start:
            image_adv = image_adv + torch.empty_like(image_adv).uniform_(-self.eps,self.eps)
            image_adv = torch.clamp(image_adv,min=0.,max=1.).detach()
        
        for _ in range(self.step):
            image_adv.requires_grad = True
            output = model(image_adv)
            loss = loss_fn(output,target)
            g = grad(loss,image_adv,retain_graph=False, create_graph=False)[0].sign()
            
            image_adv = image_adv + self.alpha*g
            delta = ( image_adv - image ).clamp_(min=-self.eps,max=self.eps).detach()
            image_adv = image + delta
            image_adv.clamp(0., 1.).detach_()
        return (image + delta).clamp(min=0.,max=1.)
            
    def __call__(self,model,x,y):
        return self.forward(model,x,y)



class FGSM():
    def __init__(self,devices, eps=8/255):
        self.eps = eps
        self.device = devices
        
    def forward(self,model,images, labels):
        r"""
        Overridden.
        """
        model.eval()
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = model(images)
        cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
    
    def __call__(self,model,x,y):
        return self.forward(model,x,y)
    
                    