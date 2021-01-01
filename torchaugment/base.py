import torch
import torchaudio
from torch.distributions.chi2 import Chi2


def to_torch_size(s):
  return torch.Size(s) if not isinstance(s, torch.Size) else s


def black(image):
  return torch.zeros_like(image)


def mean(image, grayscale=False, channel=True):
  b, c, h, w  = image.shape

  if grayscale:
    image = VF.rgb_to_grayscale(image)

  if channel:
    image = image.flatten(2).mean(-1)
    image = image.view(b,c,1,1).expand(-1,-1,h,w)
  else:
    image = image.flatten(1).mean(-1)
    image = image.view(b,1,1,1).expand(-1,c,h,w)

  return image


def blur(image):
  kernel = 1/13 * torch.tensor(
    [[1, 1, 1],
     [1, 5, 1],
     [1, 1, 1]],
    device=image.device,
    dtype=image.dtype)

  kernel = kernel.view(1,1,3,3)
  kernel = kernel.expand(3,-1,-1,-1)

  blur = F.conv2d(image, kernel, padding=1, groups=3)  
  blur = torch.clamp(blur, 0, 255)

  blur[...,(0,-1),:] = image[...,(0,-1),:]
  blur[...,:,(0,-1)] = image[...,:,(0,-1)]

  return blur


def gaussian_noise_spectogram(size, beta=0.0, sample_rate=16_000, f_min=20, f_max=None):
  f_max = sample_rate // 2 if f_max is None else f_max

  chi22 = Chi2(torch.tensor(2.0))
  size = to_torch_size(size)
  
  rank = [1] * len(size)
  rank[-2] = -1

  omega = torch.linspace(f_max, f_min, size[-2])
  law = (1 / omega ** beta).view(rank)

  noise = 0.5 * law * chi22.sample(size)
  return noise / noise.mean()


def gaussian_noise_melspectogram(size, 
                                 beta=0.0, 
                                 sample_rate=16_00, 
                                 f_min=20, 
                                 f_max=None, 
                                 n_fft=None):

  n_mels = size[-2]
  n_fft = 4 * n_mels if n_fft is None else n_fft
  size[-2] = n_fft

  noise = gaussian_noise_spectogram(size, 
                                    beta=beta, 
                                    sample_rate=sample_rate,
                                    f_min=f_min,
                                    f_max=f_max)
  
  melscale = torchaudio.transforms.MelScale(n_mels=n_mels,
                                            sample_rate=sample_rate,
                                            f_min=f_min, 
                                            f_max=f_max)

  return melscale(noise)
