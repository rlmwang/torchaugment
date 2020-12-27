import torch
import torchaugment.utils as aug_utils
import torchaugment.mask as aug_mask

from torch.distributions.beta import Beta


def mixup(image, labels, alpha=1.0):
  """Apply mixup (https://arxiv.org/abs/1710.09412).
  """
  b = image.shape[0]

  perm = torch.randperm(b)
  dist = Beta(alpha, alpha)
  lam = dist.sample([b,1])
  lam = lam.to(image.device)

  image = aug_utils.blend(image, image[perm,...], lam)
  labels = lam * labels + (1 - lam) * labels[perm,...]

  return image, labels


def cutmix(image, labels, alpha=1.0):
  """Apply cutmix (https://arxiv.org/abs/1905.04899).
  """
  b, c, h, w = image.shape

  perm = torch.randperm(b)
  dist = Beta(alpha, alpha)
  lam = dist.sample([1])
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
  dist = Beta(alpha, alpha)
  lam = dist.sample([b,1])
  lam = lam.to(image.device)

  image = aug_mask.fmix(image, lam=lam, decay=decay)
  image, mask = aug_mask.detach(image)

  image = (1 - mask) * image + mask * image[perm,...]
  labels = lam * labels + (1 - lam) * labels[perm,...]

  return image, labels


def cutmixup(image, labels, lam=0.5, alpha=1.0, masking=aug_mask.random_block):
  """Combines CutMix and MixUp, designed to work with RandAugment.
  """
  b = image.shape[0]

  perm = torch.randperm(b)
  dist = Beta(alpha, alpha)
  rho = dist.sample([1])
  rho = rho.to(image.device)

  tau = lam + rho * (1 - lam)
  sig = 1 - rho * (1 - lam)

  blend = aug_utils.blend(image, image[perm,...], tau)
  image, mask = aug_mask.detach(masking(image, lam=sig))

  image = (1 - mask) * image + mask * blend
  labels = lam * labels + (1 - lam) * labels[perm,...]

  return image, labels