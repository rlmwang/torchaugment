"""
Images should have the shape b x c x h x w.
Masks attach an alpha channel with masking values in the range [0, 1], which can
be consumed by other augmentation layers. Masks themselves consume alpha 
channels by multiplying the old with the new.
"""
import math
import torch
import torch.fft
from torch import Tensor


def to_tensor(x):
  return torch.tensor(x) if not isinstance(x, Tensor) else x


def _attach(image, mask):
  b, c, h, w = image.shape

  mask = mask.expand(b,1,h,w)
  mask = mask.to(image.device)

  if c == 3:
    mask = mask.to(image.dtype)
    return torch.cat([image, mask],1)

  elif c == 4:
    image[:,3,...] *= mask
    return image


def detach(image):
  return image[:,:3,:,:], image[:,3:,:,:]


def cutout(image, size):
  b, c, h, w = image.shape

  size_h, size_w = size
  size_h = to_tensor(size_h).to(torch.int64).to(image.device).view(-1,1,1,1)
  size_w = to_tensor(size_w).to(torch.int64).to(image.device).view(-1,1,1,1)

  center_h = torch.randint(h, (b,1,1,1), device=image.device)
  center_w = torch.randint(w, (b,1,1,1), device=image.device)

  mask_h = torch.arange(h, device=image.device).view(1,1,-1,1)
  mask_w = torch.arange(w, device=image.device).view(1,1,1,-1)

  mask = (center_h - size_h <= mask_h) & (mask_h < center_h + size_h) \
       & (center_w - size_w <= mask_w) & (mask_w < center_w + size_w)

  return _attach(image, mask)


def random_pixel(image, lam=0.5, kernel=1):
  b, c, h, w = image.shape

  h_ = h // kernel + (h % kernel != 0)
  w_ = w // kernel + (w % kernel != 0)

  rand = torch.rand([b,1,h_,w_], device=image.device)
  rand = rand.repeat_interleave(kernel, dim=2)
  rand = rand.repeat_interleave(kernel, dim=3)
  rand = rand[:,:,:h,:w]

  lam = to_tensor(lam).view(-1,1,1,1)

  return _attach(image, rand <= lam)


def random_row(image, lam=0.5, kernel=1):
  b, c, h, w = image.shape

  h_ = h // kernel + (h % kernel != 0)
    
  rand = torch.rand([b,1,h_,1], device=image.device)
  rand = rand.repeat_interleave(kernel, dim=2)
  rand = rand.expand(-1,-1,-1,w)[:,:,:h,:]
  
  lam = to_tensor(lam).view(-1,1,1,1)

  return _attach(image, rand <= lam)


def random_col(image, lam=0.5, kernel=1):
  b, c, h, w = image.shape

  w_ = w // kernel + (w % kernel != 0)
    
  rand = torch.rand([b,1,1,w_])
  rand = rand.expand(-1,-1,h,-1)[:,:,:,:w]
  
  lam = to_tensor(lam).view(-1,1,1,1)

  return _attach(image, rand <= lam)


def random_block(image, size=[50,50], lam=None):
  b, c, h, w = image.shape
  device = image.device

  if lam is not None:
    sqrt_lam = torch.sqrt(lam)
    size = (h * sqrt_lam, w * sqrt_lam)

  if size == [h,w] or all(s == [h,w] for s in size):
    return _attach(image, torch.ones(b,1,h,w))

  size_h, size_w = size
  size_h = to_tensor(size_h).to(torch.int64).to(device).view(-1,1,1,1)
  size_w = to_tensor(size_w).to(torch.int64).to(device).view(-1,1,1,1)

  rand_h = torch.floor(torch.rand([b,1,1,1], device=device) * (h - size_h + 1))
  rand_w = torch.floor(torch.rand([b,1,1,1], device=device) * (w - size_w + 1))
  
  mask_h = torch.arange(h, device=device).view(1,1,-1,1).expand(b,-1,-1,-1)
  mask_w = torch.arange(w, device=device).view(1,1,1,-1).expand(b,-1,-1,-1)
  
  mask = (rand_h <= mask_h) & (mask_h < rand_h + size_h) \
       & (rand_w <= mask_w) & (mask_w < rand_w + size_w)

  return _attach(image, mask)


def random_row_strip(image, **kwargs):
  return random_strip(image, 2, **kwargs)


def random_col_strip(image, **kwargs):
  return random_strip(image, 3, **kwargs)


def random_strip(image, dim, size=50, lam=None):
  b, c = image.shape[:2]
  d = image.shape[dim]

  device = image.device

  if lam is not None:
    size = d * lam

  size = to_tensor(size).to(device).view(-1,1,1,1)

  start = torch.rand([b,1,1,1], device=device) * (d - size)

  index = torch.arange(d, device=device).view(1,1,1,d)
  mask = (start <= index) & (index < start + size)
  mask = mask.transpose(-1,dim)

  return _attach(image, mask)


def time(image, lam=1.0):
  size = lam * image.shape[-1]
  return specaugment(image, size, -1)


def frequency(image, lam=1.0):
  size = lam * image.shape[-2]
  return specaugment(image, size, -2)


def specaugment(image, size, dim):
  b = image.shape[0]
  d = image.shape[dim]
 
  size = to_tensor(size).view(-1,1,1,1)

  width = torch.rand([b,1,1,1]) * size
  start = torch.rand([b,1,1,1]) * (d - width)

  mask = torch.arange(0,d).view([1,1,1,-1])
  mask = (start <= mask) & (mask < start + width)
  mask = mask.transpose(-1,dim)

  return _attach(image, mask)


def fmix(image, lam=None, decay=3.0):
  b, c, h, w = image.shape
  mask = low_freq_mask([b,1,h,w], decay)
  mask = binarise_mask(mask, lam)
  return _attach(image, mask)


def fftfreq(n, d=1.0, device='cpu'):
  """DFT sample frequency
  """
  s = (n - 1) // 2 + 1
  results = torch.empty(n, device=device)
  results[:s] = torch.arange(0, s, device=device)
  results[s:] = torch.arange(-(n // 2), 0, device=device)
  return results * (1.0 / (n * d))


def fftfreq2(h, w, device='cpu'):
  """Magnitude of 2d sample frequency
  """
  fy = fftfreq(h, device=device)
  fy = fy.unsqueeze(-1)

  if w % 2 == 1:
    fx = fftfreq(w, device=device)
    fx = fx[: w // 2 + 2]
  else:
    fx = fftfreq(w, device=device)
    fx = fx[: w // 2 + 1]
  
  return torch.sqrt(fx * fx + fy * fy)


def get_spectrum(shape, decay, device='cpu'):
  b, c, h, w = shape

  cap = torch.tensor(1.0 / max(h,w), device=device)
  freqs = fftfreq2(h, w, device=device)
  freqs = torch.maximum(freqs, cap)

  h, w = freqs.shape
  scale = 1.0 / (freqs ** decay).view(1,1,h,w,1)

  spec = scale * torch.randn([b,c,h,w,2])
  return spec[...,0] + spec[...,1] * 1j


def low_freq_mask(shape, decay):
  h, w = shape[-2:]

  spec = get_spectrum(shape, decay)
  mask = torch.fft.ifftn(spec, s=(h,w)).real

  lo = mask.flatten(2).min(-1)[0]
  hi = mask.flatten(2).max(-1)[0]
  lo = lo.view(shape[0],1,1,1)
  hi = hi.view(shape[0],1,1,1)

  return (mask - lo) / (hi - lo)


def binarise_mask(mask, lam):
  shape = mask.shape
  
  mask = mask.flatten(1)
  index = mask.argsort(-1, descending=True)

  if torch.rand(1) < 0.5:
    cutoff = torch.ceil(lam * mask.shape[-1])
  else:
    cutoff = torch.floor(lam * mask.shape[-1])
  cutoff = cutoff.to(torch.int64)

  for msk, idx, cut in zip(mask, index, cutoff):
    msk[idx[:cut]] = 1
    msk[idx[cut:]] = 0

  return mask.view(shape)