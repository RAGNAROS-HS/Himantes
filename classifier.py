mport torch
from datasets import load_dataset
from diffusers import DiffusionPipeline, DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from transformers import AutoencoderKL
from torch.utils.data import DataLoader
import torch.nn.functional as F


ds = load_dataset("pranavs28/pokemon_types")