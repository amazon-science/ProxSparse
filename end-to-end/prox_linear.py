import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import config

class ProxLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(ProxLinear, self).__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.original_weight = self.weight.data.clone()


    def modify_weight(self, weight):
        qw = self.sparsify_magnitude(weight, 2, 4)
        return qw


    def sparsify_magnitude(self, weight, n, m): 
        dim = weight.shape
        weight = weight.view(-1, m)
        w_mask = torch.zeros_like(weight)
        w_mask.scatter_(1, torch.topk(torch.abs(weight), n, dim = 1, largest = True)[1], True)
        weight = w_mask * weight
        weight = weight.view(dim[0], dim[1])
        return weight

    def custom_sign(self, x):
        return torch.where(x == 0, torch.tensor(1, dtype=x.dtype, device=x.device), torch.sign(x))

    def compute_proximal_loss(self):
        ratio = self.weight / (self.original_weight + config.epsilon * self.custom_sign(self.original_weight))# )
        diff = self.weight - self.original_weight
        product = ratio * diff        
        final = torch.norm(product, p=2) ** 2
        return final

    def project_mask(self):
        dim = self.weight.data.shape
        self.weight.data = self.weight.data.view(-1, 4)
        self.original_weight = self.original_weight.view(-1, 4)
        top_two_indices = torch.topk(torch.abs(self.weight.data), 2, dim=1).indices
        self.weight.data.scatter_(1, top_two_indices, self.original_weight.gather(1, top_two_indices))
        self.weight.data = self.weight.data.view(dim[0], dim[1])
        self.original_weight = self.original_weight.view(dim[0], dim[1])



def replace_prox_linear(model):
    with torch.no_grad():
        for name, m in model.named_children():
            if isinstance(m, nn.Linear) and ("bias" not in name and ("k_proj" in name or "v_proj" in name or "q_proj" in name or "o_proj" in name
                                             or "up_proj" in name or "down_proj" in name or "gate_proj" in name
                                             or "out_proj" in name or "fc1" in name or "fc2" in name)):
                newlinear = ProxLinear(m.in_features, m.out_features, m.bias is not None, device = m.weight.device, dtype=m.weight.dtype)
                newlinear.weight.data.copy_(m.weight.data)
                newlinear.original_weight.copy_(m.weight.data)
                if m.bias is not None:
                    newlinear.bias.data.copy_(m.bias.data)
                setattr(model, name, newlinear)
            elif isinstance(m, torch.nn.LayerNorm):
                pass
            else:
                replace_prox_linear(m)