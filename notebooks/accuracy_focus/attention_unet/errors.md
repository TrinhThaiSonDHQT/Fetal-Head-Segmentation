---

ImportError Traceback (most recent call last)
/tmp/ipykernel_37/1986871309.py in <cell line: 0>()
12 from shared.src.data import HC18Dataset
13 from shared.src.metrics.segmentation_metrics import dice_coefficient, iou_score, pixel_accuracy
---> 14 from shared.src.utils.visualization import save_prediction_grid, visualize_sample
15 from shared.src.utils.transforms import get_transforms
16

/kaggle/input/fhs-attention-unet/shared/src/utils/**init**.py in <module>
3 """
4
----> 5 from .transforms import get_transforms
6 from .logger import TrainingLogger
7 from .saver import PredictionSaver, save_model_weights, load_model_weights

/kaggle/input/fhs-attention-unet/shared/src/utils/transforms.py in <module>
6 """
7
----> 8 import albumentations as A
9 from albumentations.pytorch import ToTensorV2
10 import numpy as np

/usr/local/lib/python3.11/dist-packages/albumentations/**init**.py in <module>
1 **version** = "1.4.0"
2
----> 3 from .augmentations import _
4 from .core.composition import _
5 from .core.serialization import \*

/usr/local/lib/python3.11/dist-packages/albumentations/augmentations/**init**.py in <module>
6
7 # New transformations goes to individual files listed below
----> 8 from .domain_adaptation import _
9 from .dropout.channel_dropout import _
10 from .dropout.coarse_dropout import \*

/usr/local/lib/python3.11/dist-packages/albumentations/augmentations/domain_adaptation.py in <module>
4 import cv2
5 import numpy as np
----> 6 from qudida import DomainAdapter
7 from skimage.exposure import match_histograms
8 from sklearn.decomposition import PCA

/usr/local/lib/python3.11/dist-packages/qudida/**init**.py in <module>
4 import cv2
5 import numpy as np
----> 6 from sklearn.decomposition import PCA
7 from typing_extensions import Protocol
8

/usr/local/lib/python3.11/dist-packages/sklearn/**init**.py in <module>
82 \_distributor_init, # noqa: F401
83 )
---> 84 from .base import clone
85 from .utils.\_show_versions import show_versions
86

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in <module>
16 from . import **version**
17 from .\_config import config_context, get_config
---> 18 from .exceptions import InconsistentVersionWarning
19 from .utils.\_estimator_html_repr import \_HTMLDocumentationLinkMixin, estimator_html_repr
20 from .utils.\_metadata_requests import \_MetadataRequester, \_routing_enabled

ImportError: cannot import name 'InconsistentVersionWarning' from 'sklearn.exceptions' (/usr/local/lib/python3.11/dist-packages/sklearn/exceptions.py)
