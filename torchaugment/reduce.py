import torch
import torchaugment.utils as utils
import torchaugment.mask as augmask
import torchaugment.base as augbase

import torch.nn.functional as F
import torchvision.transforms.functional as VF

from torch.distributions.beta import Beta


def sharpness(image, lam=1.0, alpha=None, masking=augmask.random_block):
  """Blends image with softened version. 
  """
  b, c, h, w = image.shape

  reduced = augbase.blur(image)

  if alpha = None:
    rho = 1
  else:
    rho = Beta(alpha, alpha).sample([b,1])
    rho = rho.to(image.device)

  tau = (1 - rho) * lam + rho
  sig = (1 - rho) + rho * lam

  blend = utils.blend(image, reduced, tau)
  image, mask = augmask.detach(masking(image, lam=sig * 0.5))
  
  return (1 - mask) * image + mask * blend


def brightness(image, lam=1.0, alpha=1.0, masking=augmask.random_block): 
  # Blend image with black.
  b, c, h, w = image.shape

  reduced = augbase.black(image)

  if alpha = None:
    rho = 1
  else:
    rho = Beta(alpha, alpha).sample([b,1])
    rho = rho.to(image.device)

  tau = (1 - rho) * lam + rho
  sig = (1 - rho) + rho * lam

  blend = utils.blend(image, reduced, tau)
  image, mask = augmask.detach(masking(image, lam=sig * 0.5))
  
  return (1 - mask) * image + mask * blend


def contrast(image, lam=1.0, alpha=1.0, grayscale=False,
             masking=augmask.random_block): 
  # Blend image with its average per channel.
  b, c, h, w = image.shape

  reduced = augbase.mean(image, grayscale=grayscale)

  if alpha = None:
    rho = 1
  else:
    rho = Beta(alpha, alpha).sample([b,1])
    rho = rho.to(image.device)

  tau = (1 - rho) * lam + rho
  sig = (1 - rho) + rho * lam

  blend = utils.blend(image, reduced, tau)
  image, mask = augmask.detach(masking(image, lam=sig * 0.5))
  
  return (1 - mask) * image + mask * blend

