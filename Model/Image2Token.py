import torch.nn as nn
import torch
import numpy as np
import math

class Patch_Tokenization(nn.Module):
   def __init__(self,
                img_size: tuple[int, int, int] = (3, 64, 64),
                patch_size: int = 16,
                token_len: int = 768 # Токен сайз - берется в базовых моделях VIT
                ): #Ввести параметры
      super().__init__()

      self.img_size = img_size
      C, H, W = self.img_size

      self.patch_size = patch_size

      assert H % patch_size == 0, 'Высота изображения должна делиться нацело на patch_size'
      assert W % patch_size == 0, 'Ширина изображения должна делиться нацело на patch_size'

      self.num_tokens = (H / self.patch_size) * (W / self.patch_size)
      patch_dim = C * patch_size * patch_size

      self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size, padding=0)
      self.project = nn.Linear(patch_dim, token_len)

   def forward(self, x):
      B = x.shape[0]
      x = self.unfold(x)
      x = x.transpose(1, 2)
      x = self.project(x)
      return x


def get_sinusoid_encoding(num_tokens, token_len):
   """
   Синусоидальное позиционное кодирование
   - информация о позиции каждого патча
   """

   def get_position_angle_vec(i):
      """
      Поиск позиции
      """
      return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

   sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
   sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
   sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

   return torch.FloatTensor(sinusoid_table).unsqueeze(0)
