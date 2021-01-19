import torch
import torch.nn as nn


def to_tensor(x):
  return torch.tensor(x) if not isinstance(x, Tensor) else x


class Aug(nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__()
    self.args = args
    self.kwargs = kwargs

  def forward(self, x):
    return self.function(x, *self.args, **self.kwargs)

  def __repr__(self):
    args = ', '.join(str(x) for x in self.args)
    kwargs = ', '.join(f'{k}={self.kwargs[k]}' for k in self.kwargs) 
    inputs = ", ".join([args, kwargs]).strip(', ')
    return f'{self.__class__.__name__}({inputs})'


def blend(image1, image2, lam=0.5):
  """Blend image1 and image2 by linearly interpolating by a factor lambda.
  """
  ones = [1] * (image1.dim() - 1)

  lam = to_tensor(lam)
  lam = lam.view(-1, *ones)
  lam = lam.to(image1.dtype)
  lam = lam.to(image1.device)

  return image1 + lam * (image2 - image1)