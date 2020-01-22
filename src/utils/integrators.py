import numpy as np
import torch
import torchdiffeq
from scipy.integrate import solve_ivp
from src.utils.op import gpu_numpy_detach

