import torch
import torchaugment.utils as aug_utils
import torchaugment.mask as aug_mask

from torch.distributions.beta import Beta


def mixup(image, labels, alpha=1.0):
  """Apply mixup (https://arxiv.org/abs/1710.09412).
  """
  b = image.shape[0]

  perm = torch.randperm(b)

  lam = Beta(alpha, alpha).sample([b,1])
  lam = lam.to(image.device)

  image = aug_utils.blend(image, image[perm,...], lam)
  labels = lam * labels + (1 - lam) * labels[perm,...]

  return image, labels


def cutmix(image, labels, alpha=1.0):
  """Apply cutmix (https://arxiv.org/abs/1905.04899).
  """
  b, c, h, w = image.shape

  perm = torch.randperm(b)

  lam = Beta(alpha, alpha).sample([b,1])
  lam = lam.to(image.device)

  size = [h * torch.sqrt(1 - lam) / 2,
          w * torch.sqrt(1 - lam) / 2]

  image = aug_mask.cutout(image, size)
  image, mask = aug_mask.detach(image)

  image = (1 - mask) * image + mask * image[perm,...]
  labels = lam * labels + (1 - lam) * labels[perm,...]

  return image, labels


def fmix(image, labels, decay=3.0, alpha=1.0):
  """Apply cutmix (https://arxiv.org/abs/1905.04899).
  """
  b = image.shape[0]

  perm = torch.randperm(b)

  lam = Beta(alpha, alpha).sample([b,1])
  lam = lam.to(image.device)

  image = aug_mask.fmix(image, lam=lam, decay=decay)
  image, mask = aug_mask.detach(image)

  image = (1 - mask) * image + mask * image[perm,...]
  labels = lam * labels + (1 - lam) * labels[perm,...]

  return image, labels


def cutmixup(image, labels, lam=1.0, alpha=1.0, 
             masking=aug_mask.random_block):
  """Combines CutMix and MixUp, designed for RandAugment.
  The parameter lambda represents the intensity of augmentation (either a 50/50
  mixup or covering 50% of the image with cutmix). Whether to prioritize cutmix 
  or mixup is decided randomly from a Beta distribution with a single parameter 
  alpha. It also accepts custom masking strategies for the cutmix component.
  """
  b = image.shape[0]

  perm = torch.randperm(b)

  rho = Beta(alpha, alpha).sample([b,1])
  rho = rho.to(image.device)
  
  tau = (1 - rho) * lam * 0.5 + rho
  sig = (1 - rho) + rho * lam * 0.5

  blend = aug_utils.blend(image, image[perm,...], tau)
  image, mask = aug_mask.detach(masking(image, lam=sig))

  image = (1 - mask) * image + mask * blend
  labels = (1 - lam * 0.5) * labels + lam * 0.5 * labels[perm,...]

  return image, labels