"""
Механизм внимание
Сравнивается каждый патч со всеми другими
Для каждого токена Attention вычисляет: Q-запросы K-ключи V-значения;
Решает как патчи связаны
"""
import typing
import numpy as np
import torch
import torch.nn as nn

NoneFloat = typing.Union[None, float]

class Attention(nn.Module):
   def __init__(self,
                dim: int,
                num_heads: int=1,
                drop: float=0.1,
                qkv_bias: bool = False,
                qk_scale: NoneFloat = None):

      super().__init__()

      
      self.num_heads = num_heads
      self.chan = dim
      self.head_dim = self.chan // self.num_heads
      self.scale = qk_scale or self.head_dim ** -0.5
      assert self.chan % self.num_heads == 0, '"Chan" должен делится на num_heads'

      self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
      self.proj = nn.Linear(dim, dim)
      self.attn_drop = nn.Dropout(drop)

   def forward(self, x):
      B, N, C = x.shape
      qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
      q, k, v = qkv[0], qkv[1], qkv[2]

      attn = (q * self.scale) @ k.transpose(-2, -1)
      attn = attn.softmax(dim=-1)
      attn = self.attn_drop(attn)

      x = (attn @ v).transpose(1, 2).reshape(B, N, self.chan)

      x = self.proj(x)

      return x
