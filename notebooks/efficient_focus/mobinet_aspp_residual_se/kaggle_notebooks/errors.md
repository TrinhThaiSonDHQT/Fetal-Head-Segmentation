---

ImportError Traceback (most recent call last)
/tmp/ipykernel_48/4066336658.py in <cell line: 0>()
9
10 from efficient_focus.src.losses import DiceBCEWithLogitsLoss
---> 11 from efficient_focus.src.models.mobinet_aspp_residual_se.mobinet_aspp_residual_se import MobileNetV2ASPPResidualSEUNet, count_parameters
12
13 from shared.src.data import LargeScaleDataset

/kaggle/input/fetal-head-segmentation/efficient_focus/src/models/mobinet_aspp_residual_se/mobinet_aspp_residual_se.py in <module>
28 import torch
29 import torch.nn as nn
---> 30 import torchvision.models as models
31
32 # Use proper relative imports from parent package

/usr/local/lib/python3.11/dist-packages/torchvision/**init**.py in <module>
8 # .extensions) before entering \_meta_registrations.
9 from .extension import \_HAS_OPS # usort:skip
---> 10 from torchvision import \_meta_registrations, datasets, io, models, ops, transforms, utils # usort:skip
11
12 try:

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/**init**.py in <module>
----> 1 from .\_optical_flow import FlyingChairs, FlyingThings3D, HD1K, KittiFlow, Sintel
2 from .\_stereo_matching import (
3 CarlaStereo,
4 CREStereo,
5 ETH3DStereo,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/\_optical_flow.py in <module>
10 from PIL import Image
11
---> 12 from ..io.image import decode_png, read_file
13 from .utils import \_read_pfm, verify_str_arg
14 from .vision import VisionDataset

/usr/local/lib/python3.11/dist-packages/torchvision/io/**init**.py in <module>
16 VideoMetaData,
17 )
---> 18 from .image import (
19 decode_avif,
20 decode_gif,

/usr/local/lib/python3.11/dist-packages/torchvision/io/image.py in <module>
6
7 from ..extension import \_load_library
----> 8 from ..utils import \_log_api_usage_once
9
10

/usr/local/lib/python3.11/dist-packages/torchvision/utils.py in <module>
9 import numpy as np
10 import torch
---> 11 from PIL import Image, ImageColor, ImageDraw, ImageFont
12
13

/usr/local/lib/python3.11/dist-packages/PIL/ImageDraw.py in <module>
37 from typing import cast
38
---> 39 from . import Image, ImageColor, ImageText
40
41 TYPE_CHECKING = False

/usr/local/lib/python3.11/dist-packages/PIL/ImageText.py in <module>
2
3 from . import ImageFont
----> 4 from .\_typing import \_Ink
5
6

ImportError: cannot import name '\_Ink' from 'PIL.\_typing' (/usr/local/lib/python3.11/dist-packages/PIL/\_typing.py)
