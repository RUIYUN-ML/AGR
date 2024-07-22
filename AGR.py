import torch

class OGD():
    def __init__(self) :
        pass
    def get_gradient(self,optim,loss,model):
        """
        Compute the gradient,return the gradient as a one-dimensional vector
        """
        grad_list = []
        params = list(model.parameters())
        optim.zero_grad()
        grad = torch.autograd.grad(loss,params,retain_graph=True)
        for g in grad:
            grad_list.append(g.view(-1))
        return torch.cat(grad_list)     
    
    def get_cos_similarity(self,clean_gradient,adv_gradient):
        flat_norm1 = clean_gradient
        flat_norm2 = adv_gradient
        norm1 = torch.norm(flat_norm1,2)
        norm2 = torch.norm(flat_norm2,2)
        dot_product = torch.dot(flat_norm1, flat_norm2)
        similarity = dot_product / (norm1 * norm2 +1E-20)
        return similarity.item()
    
    def get_l2_distance(self,clean_gradient,adv_gradient):
        flat_norm1 = clean_gradient
        flat_norm2 = adv_gradient
        norm_dis = torch.norm(flat_norm1 - flat_norm2 , p=2)
        return norm_dis.item()