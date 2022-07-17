from datetime import datetime
from vncorenlp import VnCoreNLP
from .helper import init_logger

import os

LOGGER = init_logger(datetime.now().strftime('%d%b%Y_%H-%M-%S.log'))

ASPECT_LABELS = ["O", "B-SCREEN", "B-CAMERA", "B-FEATURES", "B-BATTERY", "B-PERFORMANCE", "B-STORAGE", "B-DESIGN", "B-PRICE",
                 "B-GENERAL", "B-SER&ACC", "I-SCREEN", "I-CAMERA", "I-FEATURES", "I-BATTERY", "I-PERFORMANCE",
                 "I-STORAGE", "I-DESIGN", "I-PRICE", "I-GENERAL", "I-SER&ACC"]

POLARITY_LABELS = ["O", "B-NEGATIVE", "I-NEGATIVE", "B-NEUTRAL", "I-NEUTRAL", "B-POSITIVE", "I-POSITIVE"]

if os.path.exists("vncorenlp/VnCoreNLP-1.1.1.jar"):
    RDRSEGMENTER = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
else:
    RDRSEGMENTER = None
