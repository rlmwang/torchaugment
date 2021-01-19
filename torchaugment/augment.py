import functools
import numbers

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF

from torch.distributions.beta import Beta
from copy import deepcopy

from torchaugment.utils import Aug


def uint8_histc(image):
  b, c, = image.shape[:2]
  try:
    image = image.view(b,c,-1,1)
  except:
    image = image.reshape(b,c,-1,1)
  bins = torch.arange(0, 256).view(1,1,1,-1)
  return (image == bins).sum(-2)


def randaug(level_policy):
  """Adds a key word argument 'level' to the augmentation function.
  The level policy is function that translates a level value to 
  appropriate arguments of the augmentation.
  """
    
  def decorator(aug):
    
    @functools.wraps(aug)
    def wrapper(image, *args, **kwargs):

      if 'level' in kwargs:
        args, kwargs = level_policy(image, kwargs['level'])
        return aug(image, *args, **kwargs)
    
      else:
        return aug(image, *args, **kwargs)
    
    return wrapper
  return decorator


def replaces(aug):
  """Augmentations such as skew fill empty space with zeros.
  This decorator replaces those zeroes with a replacement value.
  The replacement value is added as key word argument.
  """
    
  @functools.wraps(aug)
  def wrapper(image, *args, replace='channel', **kwargs):
    replace = _parse_replace(replace, image)
    
    image = _wrap(image)
    image = aug(image, *args, **kwargs)
    return _unwrap(image, replace)

  return wrapper


def _parse_replace(replace, image):
  if replace == 'channel':
    return _replace_channel(image)
  if replace == 'grayscale':
    return _replace_grayscale(image)
  return replace


def _replace_channel(image):
  replace = image.flatten(2).to(torch.float32).mean(-1)
  return replace.squeeze()


def _replace_grayscale(image):
  image = VF.rgb_to_grayscale(image)
  replace = image.flatten(2).to(torch.float32).mean(-1)
  return replace.squeeze()


def _wrap(image):
  """Returns image with an extra channel set to all 1s.
  """
  ones = torch.ones_like(image[:,:1,:,:])
  return torch.cat([image, ones], 1)


def _unwrap(image, replace):
  """Unwraps an image produced by wrap by filling each channel 
  with the replacement values wherever the wrapper channel is zero.
  """
  image, alpha = image[:,:3,...], image[:,3:4,...]

  b, c, h, w = image.shape
  
  replace = replace.to(image.dtype)
  replace = replace.to(image.device)

  if replace.dim() == 2:
    replace = replace.view(-1,c,1,1)
  else:
    replace = replace.view(-1,1,1,1)

  return torch.where(alpha == 1, image, replace)


def _to_tensor(x, dtype=None, device=None):
  x = torch.tensor(x) if not isinstance(x, Tensor) else x
  x = x.to(dtype) if dtype is not None else x
  x = x.to(device) if device is not None else x
  return x


def _to_camel_case(snake_str):
  return ''.join(x.title() for x in snake_str.split('_'))


class RandomCrop(torch.nn.Module):
  """
  Adopted from pytorch 1.7, added iid_crops.
  """
  @staticmethod
  def get_params(img, output_size, iid_crops):
    b, c, h, w = img.shape
    th, tw = output_size

    if w == tw and h == th:
      return 0, 0, h, w

    if iid_crops:
      i = torch.randint(0, h - th + 1, size=(b,))
      j = torch.randint(0, w - tw + 1, size=(b,))
    else:
      i = torch.randint(0, h - th + 1, size=(1,)).item()
      j = torch.randint(0, w - tw + 1, size=(1,)).item()
    return i, j, th, tw

  def __init__(self, size, iid_crops=False):
    super().__init__()

    self.size = tuple(_setup_size(
        size, error_msg="Please provide only two dimensions (h,w) for size."
    ))
    self.iid_crops = iid_crops

  def forward(self, img):
    i, j, h, w = self.get_params(img, self.size, self.iid_crops)    
    return image_crop(img, i, j, h, w)

  def __repr__(self):
    return f'{self.__class__.__name__}(size={self.size})'



def image_crop(img, i, j, th, tw):
  b, c, h, w = img.shape
 
  i = _to_tensor(i, torch.int64, img.device).view(-1,1,1,1)
  j = _to_tensor(j, torch.int64, img.device).view(-1,1,1,1)
  
  if i.numel() == 1:
    i = i.expand(b,c,1,w)
  else:
    i = i.expand(-1,c,1,w)
  i = i.to(img.device)
  i = i.to(torch.int64)  
  
  if j.numel() == 1:
    j = j.expand(b,c,th,1)
  else:
    j = j.expand(-1,c,th,1)

  th = torch.arange(0,th)
  th = th.view(1,1,-1,1)
  th = th.to(img.device)

  tw = torch.arange(0,tw)
  tw = tw.view(1,1,1,-1)
  tw = tw.to(img.device)

  img = torch.gather(img, -2, i + th)
  img = torch.gather(img, -1, j + tw)
  return img


def _setup_size(size, error_msg):
  if isinstance(size, numbers.Number):
    return int(size), int(size)
  if isinstance(size, Sequence) and len(size) == 1:
    return size[0], size[0]
  if len(size) != 2:
    raise ValueError(error_msg)
  return size


def __identity__(image, level):
  return (), {}


@randaug(__identity__)
def identity(image):
  return image


class Identity(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = identity


def __cutout__(image, level):
  h, w = image.shape[-2:]
  factor = 0.2 * min(h,w)
  return (int(level * factor),), {}


@replaces
@randaug(__cutout__)
def cutout(image, size):
  """Apply cutout (https://arxiv.org/abs/1708.04552).
  This operation applies a (2*size, 2*size) mask with the given
  replace value at a uniformly random location on the image.
  """
  b, c, h, w = image.shape

  center_h = torch.randint(h, (b,1,1,1)).to(image.device)
  center_w = torch.randint(w, (b,1,1,1)).to(image.device)

  mask_h = torch.arange(h).view(1,1,-1,1).to(image.device)
  mask_w = torch.arange(w).view(1,1,1,-1).to(image.device)
  mask = (center_h - size <= mask_h) & (mask_h < center_h + size) \
       & (center_w - size <= mask_w) & (mask_w < center_w + size)

  return image.masked_fill(mask, 0)


class Cutout(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = cutout


def __solarize__(image, level):
  return (), {'threshold': int(level * 255)}


@randaug(__solarize__)
def solarize(image, threshold=128):
  """Invert all values above the threshold."""
  return torch.where(threshold <= image, image, 255 - image)


class Solarize(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = solarize


def __solarize_add__(image, level, alpha=0.4):
    
  dist = Beta(alpha, alpha)
  rand = dist.sample(image.shape[:1])
  rand = 2 * rand - 1

  constant = (rand * level * 110)
  constant = constant.to(torch.int64)
  constant = constant.to(image.device)
  constant = constant.view(-1,1,1,1)

  return (constant,), {}


@randaug(__solarize_add__)
def solarize_add(image, constant, threshold=128):
  """Add a constant to all values below the threshold.
  The constant should be between -128 and 128.
  """
  added_image = image.to(torch.int64) + constant
  added_image = torch.clamp(added_image, 0, 255)
  return torch.where(image >= threshold, image, added_image)


class SolarizeAdd(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = solarize_add


def __color__(image, level, alpha=0.4):

  dist = Beta(alpha, alpha)
  rand = dist.sample(image.shape[:1])
  rand = rand.to(image.device)
  rand = 2 * rand - 1

  return (1.0 + rand * level * 0.8,), {}


@randaug(__color__)
def color(image, factor):
  # Blend image with its grayscale version.
  grayscale = VF.rgb_to_grayscale(image, num_output_channels=3)
  return blend(grayscale, image, factor)


class Color(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = color


def __contrast__(image, level, alpha=0.4):

  dist = Beta(alpha, alpha)
  rand = dist.sample(image.shape[:1])
  rand = rand.to(image.device)
  rand = 2 * rand - 1

  return (1.0 + rand * level * 0.8,), {}


@randaug(__contrast__)
def contrast(image, factor): 
  # Blend image with its average grayscale histogram.
  b, c, h, w = image.shape

  grayscale = VF.rgb_to_grayscale(image)

  flat = grayscale.flatten(1).mean(-1) + 0.5
  flat = torch.clamp(flat, 0, 255)
  flat = flat.view(-1,1,1,1).expand(-1,c,h,w)

  return blend(flat, image, factor)


class Contrast(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = contrast


def __brightness__(image, level, alpha=0.4):

  dist = Beta(alpha, alpha)
  rand = dist.sample(image.shape[:1])
  rand = rand.to(image.device)
  rand = 2 * rand - 1

  return (1.0 + rand * level * 0.8,), {}


@randaug(__brightness__)
def brightness(image, factor):
  # Blend image with a black image.
  return blend(torch.zeros_like(image), image, factor)


class Brightness(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = brightness


def __posterize__(image, level):
  return (int(8 - level * 8),), {}


@randaug(__posterize__)
def posterize(image, bits):
  # Reduce the number of bits for each pixel.
  dtype = image.dtype
  image = image.to(torch.uint8)

  shift = max(0, 8 - bits)  
  image = (image >> shift) << shift

  return image.to(dtype)


class Posterize(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = posterize


def __rotate__(image, level, alpha=0.4):
    
  dist = Beta(alpha, alpha)
  rand = dist.sample([1])
  rand = 2 * rand - 1

  return (float(rand * level * 30),), {}


@replaces
@randaug(__rotate__)
def rotate(image, degrees):
  return VF.rotate(image, degrees)


class Rotate(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = rotate


def __translate__(image, level, alpha=0.4):
  h, w = image.shape[-2:]
  factor = 0.5 * min(h,w)

  dist = Beta(alpha, alpha)
  rand = dist.sample([1])
  rand = rand.to(image.device)
  rand = 2 * rand - 1

  return (rand * level * factor,), {}


@replaces
@randaug(__translate__)
def translate_x(image, shift):
  return VF.affine(image, 0, [shift,0], 1, 0)


@replaces
@randaug(__translate__)
def translate_y(image, shift):
  return VF.affine(image, 0, [0,shift], 1, 0)


class TranslateX(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = translate_x


class TranslateY(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = translate_y


def __shear__(image, level, alpha=0.4):

  dist = Beta(alpha, alpha)
  rand = dist.sample([1])
  rand = rand.to(image.device)
  rand = 2 * rand - 1

  return (rand * level * 30,), {}


@replaces
@randaug(__shear__)
def shear_x(image, degrees, replace=0):
  return VF.affine(image, 0, (0,0), 1, [degrees,0])


@replaces
@randaug(__shear__)
def shear_y(image, degrees, replace=0):
  return VF.affine(image, 0, (0,0), 1, [0,degrees])


class ShearX(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = shear_x


class ShearY(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = shear_y


def __auto_contrast__(image, level):
  return (level * 0.2,), {}


@randaug(__auto_contrast__)
def auto_contrast(image, cutoff, grayscale=True):
  """Implements autocontrast from PIL using torch ops.
  """
  w, h = image.shape[-2:]

  if grayscale:
    reference = VF.rgb_to_grayscale(image)
  else:
    reference = image

  hist = uint8_histc(reference)
  hist = hist.cumsum(-1)
  hist = hist / hist[...,-1:]

  if cutoff:
    lo = (hist <= cutoff).sum(-1)
    hi = 256.0 - (hist >= 1 - cutoff).sum(-1)
  else:
    lo = (hist == 0).sum(-1)
    hi = 256.0 - (hist == 1).sum(-1)

  lo = lo[:,:,None,None]
  hi = hi[:,:,None,None]
    
  scale = 255.0 / (hi - lo)
  offset = - lo * scale

  scale = scale.expand(-1,-1,w,h)
  offset = offset.expand(-1,-1,w,h)

  scaled = image * scale + offset
  scaled = torch.clamp(scaled, 0.0, 255.0)

  return image.masked_scatter(hi > lo, scaled)  


class AutoContrast(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = auto_contrast


def __sharpness__(image, level, alpha=0.4):

  dist = Beta(alpha, alpha)
  rand = dist.sample(image.shape[:1])
  rand = rand.to(image.device)
  rand = 2 * rand - 1

  return (1.0 + rand * level * 0.8,), {}


@randaug(__sharpness__)
def sharpness(image, factor):
  """Blends image with softened version. 
  """
  kernel = 1/13 * torch.tensor(
    [[1, 1, 1],
     [1, 5, 1],
     [1, 1, 1]],
    device=image.device,
    dtype=torch.float32)

  kernel = kernel.view(1,1,3,3)
  kernel = kernel.expand(3,-1,-1,-1)

  soft = F.conv2d(image, kernel, padding=1, groups=3)  
  soft = torch.clamp(soft, 0, 255)

  soft[...,(0,-1),:] = image[...,(0,-1),:]
  soft[...,:,(0,-1)] = image[...,:,(0,-1)]
  
  return blend(soft, image, factor)


class Sharpness(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = sharpness


def __equalize__(image, level):
  return (), {}


@randaug(__equalize__)
def equalize(image, grayscale=True):
  """Equalize.
  """
  b, c, h, w = image.shape

  if grayscale:
    reference = VF.rgb_to_grayscale(image)
  else:
    reference = image

  hist = uint8_histc(reference)

  index = torch.arange(256).view(1,1,-1)
  index = index.to(image.device)

  index = (hist != 0) * index
  index = torch.argmax(index, -1, keepdim=True)

  lastval = torch.gather(hist, -1, index)

  step = hist.sum(-1, keepdim=True)
  step = (step - lastval) // 225

  zeros = torch.zeros_like(hist[...,:1])
    
  lut = (hist.cumsum(-1) + step // 2) // step
  lut = torch.cat([zeros, lut[...,:-1]], -1)
  lut = torch.clamp(lut,0,255)
  lut = lut.expand(-1,3,-1) if grayscale else lut

  dtype = image.dtype

  image = image.reshape(b,c,-1).to(torch.int64)
  image = torch.gather(lut, -1, image).view(b,c,h,w)

  return image.to(dtype)


class Equalize(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = equalize


def __invert__(image, level):
  return (), {}


@randaug(__invert__)
def invert(image):
  return 255 - image


class Invert(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = invert


def __gaussian_noise__(image, level):
  return (level * 32,), {}


@randaug(__gaussian_noise__)
def gaussian_noise(image, std):
  noise = std * torch.randn(image.size())
  noise = noise.to(image.device)

  return torch.clamp(image + noise, 0, 255)


class GaussianNoise(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = gaussian_noise


def __mask_time__(image, level):
  w = image.shape[-1]
  return (level * 0.4 * w,), {}


def __mask_frequency__(image, level):
  h = image.shape[-2]
  return (level * 0.4 * h,), {}


@replaces
@randaug(__mask_time__)
def mask_time(image, param):
  return mask_along_axis_iid(image, param, -1)


@replaces
@randaug(__mask_frequency__)
def mask_frequency(image, param):
  return mask_along_axis_iid(image, param, -2)


def mask_along_axis_iid(img, param, axis, replace=0):
  batch = img.shape[0]
  length = img.shape[axis]
    
  ones = (img.dim() - 1) * [1]
    
  start = torch.rand([batch] + ones, device=img.device)
  final = torch.rand([batch] + ones, device=img.device)

  start = start * (length - param)
  final = start + final * param
    
  mask = torch.arange(0, length, device=img.device, dtype=img.dtype)
  mask = mask.view(ones + [-1])
  mask = (start <= mask) & (mask < final)
  
  img = img.transpose(axis, -1)
  img = img.masked_fill(mask, replace)
  img = img.transpose(axis, -1)
    
  return img


class MaskTime(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = mask_time


class MaskFrequency(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = mask_frequency


def __warp_time__(spec, level):
  w = spec.shape[-1]
  return (level * 0.4 * w,), {}


def __warp_frequency__(spec, level):
  w = spec.shape[-1]
  return (level * 0.4 * w,), {}


@randaug(__warp_time__)
def warp_time(spec, param):
  return warp_along_axis(spec, param, -1)


@randaug(__warp_frequency__)
def warp_frequency(spec, param):
  return warp_along_axis(spec, param, -2)


def warp_along_axis(spec, param, axis):
  """Time warping from (https://arxiv.org/pdf/1912.05533.pdf)
  """
  spec = spec.transpose(axis,-1)
  b, c, h, w = spec.shape

  w0 = torch.rand([b] + 2 * [1], device=spec.device)
  w1 = torch.rand([b] + 2 * [1], device=spec.device)

  w0 = param + w0 * (w - 2 * param)
  w1 = param * (2 * w1 - 1)

  h_, w_ = torch.meshgrid(torch.arange(h), torch.arange(w))
  h_, w_ = h_.to(spec.device), w_.to(spec.device)
    
  h_ = h_.expand(b,-1,-1)

  w_lo = (w0 / (w0 + w1)) * w_
  w_hi = ((w - 1 - w0) * w_ - (w - 1) * w1) / (w - 1 - w0 - w1)

  w_ = torch.where(w_ <= w0 + w1, w_lo, w_hi)

  grid = torch.stack([
      2 * w_ / w - 1,
      2 * h_ / h - 1,
  ], -1)

  grid = grid.to(torch.float32).to(spec.device)
  spec = spec.to(torch.float32)
  spec = F.grid_sample(spec, grid, align_corners=False)
  spec = torch.clamp(spec,0,255)

  return spec.transpose(axis,-1)


class WarpTime(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = warp_time


class WarpFrequency(Aug):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.function = warp_frequency


AUG_LIST = [
  auto_contrast,
  equalize,
  invert,
  rotate,
  posterize,
  solarize,
  solarize_add,
  color,
  contrast,
  brightness,
  sharpness,
  shear_x,
  shear_y,
  translate_x,
  translate_y,
  cutout,
]


AUG_LIST_SMOOTH = [
  invert,
  rotate,
  solarize,
  solarize_add,
  color,
  contrast,
  brightness,
  sharpness,
  shear_x,
  shear_y,
  translate_x,
  translate_y,
  cutout,
]


SPEC_AUG_LIST = [
  mask_time,
  mask_frequency,
  warp_time,
]


def to_tensor(x):
  return torch.tensor(x) if not isinstance(x, torch.Tensor) else x


def force_shape_2d(x, shape):
  if x.numel() == 1:
    x = x.expand(shape)
  else:
    x = x.view(-1,shape[1])
    x = x[:shape[0],:]
    x = x.expand(shape[0],-1)
  return x


def rand_augment(image, num_augs, level, alpha=None,
                 augs=AUG_LIST, add_identity=True, add_augs=None):
  """RandAugment with sampling (https://arxiv.org/abs/1909.13719).
  """
  augs = deepcopy(augs)

  if add_identity:
    augs.append(identity)

  if add_augs is not None:
    augs.extend(add_augs)

  if alpha is None:
        
    for _ in range(num_augs):
      index = torch.randint(len(augs), [1])
      image = augs[index](image, level=level)

  else:
    alpha = to_tensor(alpha)
    level = to_tensor(level)
    
    alpha = force_shape_2d(alpha, [num_augs, len(augs)])
    level = force_shape_2d(level, [num_augs, len(augs)])
    
    alpha = F.softmax(alpha,-1).cumsum(-1)

    for a, l in zip(alpha, level):
      index = a <= torch.rand(1)
      index = index.sum()
      image = augs[index](image, level=l[index])
    
  return image


def rand_augment_softmax(image, alpha, level,
                         augs=AUG_LIST_SMOOTH, add_identity=True, add_augs=None):
  """RandAugment with softmax (https://arxiv.org/abs/1909.13719).
  Uses softmax(alpha) linear comibnation instead randomly sampling ops for
  Differentiable Architecture Search (https://arxiv.org/abs/1806.09055).
  """
  augs = deepcopy(augs)

  if add_identity:
    augs.append(identity)
    
  if add_augs is not None:
    augs.extend(add_augs)

  alpha = force_shape_2d(alpha, [num_augs, len(augs)])
  level = force_shape_2d(level, [num_augs, len(augs)])

  alpha = F.softmax(alpha, -1)
    
  for a, l in zip(alpha, level):
    image = [aug(image, level=l[k]) for k, aug in enumerate(augs)]
    image = torch.stack(image, -1)
    image = (a * image).sum(-1)
    
  return image