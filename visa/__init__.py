__version__ = '0.0.0'

from visa.constants import LOGGER
from visa.trainer import train as TRAIN, test as TEST
from visa.trainer_with_ray_tune import main as TRAIN_WITH_RAY

__all__ = ['TRAIN', 'TEST', 'LOGGER', 'TRAIN_WITH_RAY']