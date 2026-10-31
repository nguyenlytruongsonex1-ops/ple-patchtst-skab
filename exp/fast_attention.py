"""Equivalent fused attention for a single encoder layer; no masks or attention dropout."""
import types
import torch

def fused_forward(self,q,k,v,prev=None,key_padding_mask=None,attn_mask=None):
    assert prev is None and key_padding_mask is None and attn_mask is None
    assert self.attn_dropout.p == 0 and not self.scale.requires_grad
    output=torch.nn.functional.scaled_dot_product_attention(q,k.transpose(-2,-1),v,dropout_p=0.0,scale=float(self.scale))
    return (output,None,None) if self.res_attention else (output,None)

def enable(model):
    for layer in model.net.backbone.encoder.layers:
        layer.self_attn.sdp_attn.forward=types.MethodType(fused_forward,layer.self_attn.sdp_attn)
