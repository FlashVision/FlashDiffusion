from flashdiffusion.schedulers.base import BaseScheduler
from flashdiffusion.schedulers.ddpm import DDPMScheduler
from flashdiffusion.schedulers.ddim import DDIMScheduler
from flashdiffusion.schedulers.dpm_plus import DPMPlusPlusScheduler
from flashdiffusion.schedulers.euler import EulerScheduler, EulerAncestralScheduler
from flashdiffusion.schedulers.lcm import LCMScheduler

SCHEDULER_MAP = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "dpm++": DPMPlusPlusScheduler,
    "dpm_plus": DPMPlusPlusScheduler,
    "euler": EulerScheduler,
    "euler_a": EulerAncestralScheduler,
    "lcm": LCMScheduler,
}

__all__ = [
    "BaseScheduler",
    "DDPMScheduler", "DDIMScheduler", "DPMPlusPlusScheduler",
    "EulerScheduler", "EulerAncestralScheduler", "LCMScheduler",
    "SCHEDULER_MAP",
]
