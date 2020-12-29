import torch
import torchaugment.utils as utils
import torchaugment.mask as augmask

import torch.nn.functional as F
import torchvision.transforms.functional as VF

from torch.distributions.beta import Beta


def sharpness(image, lam=1.0, alpha=1.0, masking=augmask.random_block):
  """Blends image with softened version. 
  """
  b, c, h, w = image.shape

  kernel = 1/13 * torch.tensor(
    [[1, 1, 1],
     [1, 5, 1],
     [1, 1, 1]],
    device=image.device,
    dtype=torch.float32)

  kernel = kernel.view(1,1,3,3)
  kernel = kernel.expand(3,-1,-1,-1)

  reduced = F.conv2d(image, kernel, padding=1, groups=3)  
  reduced = torch.clamp(reduced, 0, 255)

  reduced[...,(0,-1),:] = image[...,(0,-1),:]
  reduced[...,:,(0,-1)] = image[...,:,(0,-1)]

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

  reduced = torch.zeros_like(image)

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

  reduced = VF.rgb_to_grayscale(image) if grayscale else image
  reduced = reduced.flatten(2).mean(-1) + 0.5
  reduced = torch.clamp(reduced, 0, 255)
  reduced = reduced.view(b,c,1,1).expand(-1,-1,h,w)

  rho = Beta(alpha, alpha).sample([b,1])
  rho = rho.to(image.device)  

  tau = (1 - rho) * lam + rho
  sig = (1 - rho) + rho * lam

  blend = utils.blend(image, reduced, tau)
  image, mask = augmask.detach(masking(image, lam=sig * 0.5))
  
  return (1 - mask) * image + mask * blend

