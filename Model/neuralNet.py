"""
двухслойная нейросеть
обрабатывает каждый токен отдельно
позволяет учить более сложные зависимости
"""
import typing
import torch.nn as nn

NoneFloat = typing.Union[None, float]

class NeuaralNet(nn.Module):
   def __init__(self,
                in_chan: int,
                hidden_chan: NoneFloat=None,
                out_chan: NoneFloat=None,
                activate_layer = nn.GELU,
                drop: float=0.1
                ):
      super().__init__()

      hidden_chan = hidden_chan or in_chan
      out_chan = out_chan or in_chan

      assert isinstance(hidden_chan, int) or hidden_chan.is_integer(), "Hidden channels in Neural Network module must be an integer"
      in_chan = int(in_chan)
      hidden_chan = int(hidden_chan)
      out_chan = int(out_chan)

      self.fc1 = nn.Linear(in_chan, hidden_chan)
      self.activate = activate_layer()
      self.drop = nn.Dropout(drop)
      self.fc2 = nn.Linear(hidden_chan, out_chan)

   def forward(self, x):
      x = self.fc1(x)
      x = self.activate(x)
      x = self.drop(x)
      x = self.fc2(x)
      return x
