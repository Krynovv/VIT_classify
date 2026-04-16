import torch
import torch.nn as nn
import numpy as np
import typing
import timm

from Model.Image2Token import get_sinusoid_encoding, Patch_Tokenization
from Model.attention import Attention
from Model.neuralNet import NeuaralNet

NoneFloat = typing.Union[None, float]

class Encoding(nn.Module):
   def __init__(self,
                dim: int,
                num_heads: int=1,
                hidden_chan_mul: float=4.,
                qkv_bias: bool=False,
                qk_scale: NoneFloat=None,
                activate_layer=nn.GELU,
                norm_layer=nn.LayerNorm):
      super().__init__()

      self.norm1 = norm_layer(dim)
      self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
      self.norm2 = norm_layer(dim)
      self.neuralnet = NeuaralNet(in_chan=dim,
                               hidden_chan=int(dim*hidden_chan_mul),
                               out_chan=dim,
                               activate_layer=activate_layer)

   def forward(self, x):
      x = x + self.attn(self.norm1(x))
      x = x + self.neuralnet(self.norm2(x))
      return x


class VIT_Backbone(nn.Module):
   def __init__(self,
                preds: int=200,
                token_len: int=768,
                num_tokens: int=500,
                num_heads: int=1,
                Encoding_hidden_chan_mul: float=4.,
                depth: int=12,
                qkv_bias=False,
                qk_scale=None,
                activate_layer=nn.GELU,
                norm_layer=nn.LayerNorm):
      super().__init__()

      self.token_len = token_len
      self.num_tokens = num_tokens
      self.num_heads = num_heads
      self.Encoding_hidden_chan_mul = Encoding_hidden_chan_mul
      self.depth = depth

      self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_len))
      self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(num_tokens=int(self.num_tokens+1), token_len=int(self.token_len)), requires_grad=False)

      self.blocks = nn.ModuleList([Encoding(dim=self.token_len,
                                             num_heads=self.num_heads,
                                             hidden_chan_mul = self.Encoding_hidden_chan_mul,
                                             qkv_bias = qkv_bias,
                                             qk_scale = qk_scale,
                                             acivate_layer = activate_layer,
                                             norm_layer = norm_layer)
         for i in range (self.depth)])

      self.norm = norm_layer(self.token_len)
      self.head = nn.Linear(self.token_len, preds)

      timm.layers.trunc_normal_(self.cls_token, stf=.02)

   def forward(self, x):
      B = x.shape[0]
      x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
      x = x + self.pos_embed
      for blk in self.blocks:
         x = blk(x)

      x.self.norm(x)
      x = self.head(x[:, 0])
      return x


class VIT_Model(nn.Module):
   def __init__(self,
                img_size: tuple[int, int, int]=(3, 512, 512),
                patch_size: int=64,
                token_len: int=768,
                preds: int=1,
                num_heads: int=1,
                Encoding_hidden_chan_mul: float=4.,
                depth: int=12,
                qkv_bias=False,
                qk_scale=None,
                activate_layer=nn.GELU,
                norm_layer=nn.LayerNorm):
      super().__init__()

      self.img_size = img_size
      C, H, W = self.img_size
      self.patch_size = patch_size
      self.token_len = token_len
      self.num_heads = num_heads
      self.Encoding_hidden_chan_mul = Encoding_hidden_chan_mul
      self.depth = depth

      self.patch_tokens = Patch_Tokenization(img_size, patch_size, token_len)
      num_tokens = self.patch_tokens.num_tokens

      self.backbone = VIT_Backbone(preds,
                                   self.token_len,
                                   num_tokens,
                                   self.num_heads,
                                   self.Encoding_hidden_chan_mul,
                                   self.depth,
                                   qkv_bias,
                                   qk_scale,
                                   activate_layer,
                                   norm_layer)
      self.apply(self.__init__weights)

   def _init__weights(self, w):
      if isinstance(w, nn.Linear):
         timm.layers.trunc_normal_(w.weight, stf=.02)
         if isinstance(w, nn.Linear) and w.bias is not None:
            nn.init.constant_(w.bias, 0)

      elif isinstance(w, nn.LinearNorm):
         nn.init.constant_(w.weight, 1.0)
         nn.init.constant_(w.bias, 0)

   @torch.jit.ignore
   def no_weight_decay(self):
      return {'cls_token'}

   def forward(self, x):
      x = self.patch_tokens(x)
      x = self.backbone(x)
      return (x)





