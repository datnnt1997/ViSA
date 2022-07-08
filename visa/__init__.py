__version__ = '0.0.0'

from visa.constants import LOGGER
from visa.trainer import train as TRAIN, test as TEST


__all__ = ['TRAIN', 'TEST', 'LOGGER']