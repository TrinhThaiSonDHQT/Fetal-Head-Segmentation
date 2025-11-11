---

ValueError Traceback (most recent call last)
/tmp/ipykernel_48/4066336658.py in <cell line: 0>()
13 from shared.src.data import LargeScaleDataset
14 from shared.src.metrics.segmentation_metrics import dice_coefficient, iou_score, pixel_accuracy
---> 15 from shared.src.utils.visualization import save_prediction_grid, visualize_sample
16 from shared.src.utils.transforms import get_transforms
17

/kaggle/input/fetal-head-segmentation/shared/src/utils/**init**.py in <module>
3 """
4
----> 5 from .transforms.transforms import get_transforms
6 from .transforms.aggressive_transforms import get_aggressive_transforms
7 from .logger import TrainingLogger

/kaggle/input/fetal-head-segmentation/shared/src/utils/transforms/**init**.py in <module>
3 """
4
----> 5 from .transforms import get_transforms
6 from .aggressive_transforms import get_aggressive_transforms, get_medium_transforms
7

/kaggle/input/fetal-head-segmentation/shared/src/utils/transforms/transforms.py in <module>
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
1 # Common classes
----> 2 from .blur.functional import _
3 from .blur.transforms import _
4 from .crops.functional import _
5 from .crops.transforms import _

/usr/local/lib/python3.11/dist-packages/albumentations/augmentations/blur/**init**.py in <module>
----> 1 from .functional import _
2 from .transforms import _

/usr/local/lib/python3.11/dist-packages/albumentations/augmentations/blur/functional.py in <module>
7
8 from albumentations.augmentations.functional import convolve
----> 9 from albumentations.augmentations.geometric.functional import scale
10 from albumentations.augmentations.utils import (
11 \_maybe_process_in_chunks,

/usr/local/lib/python3.11/dist-packages/albumentations/augmentations/geometric/**init**.py in <module>
----> 1 from .functional import _
2 from .resize import _
3 from .rotate import _
4 from .transforms import _

/usr/local/lib/python3.11/dist-packages/albumentations/augmentations/geometric/functional.py in <module>
5 import numpy as np
6 import skimage.transform
----> 7 from scipy.ndimage import gaussian_filter
8
9 from albumentations.augmentations.utils import (

/usr/local/lib/python3.11/dist-packages/scipy/ndimage/**init**.py in <module>
154 # mypy: ignore-errors
155
--> 156 from .\_support_alternative_backends import \*
157
158 # adjust **all** and do not leak implementation details

/usr/local/lib/python3.11/dist-packages/scipy/ndimage/\_support_alternative_backends.py in <module>
5
6 import numpy as np
----> 7 from .\_ndimage_api import \* # noqa: F403
8 from . import \_ndimage_api
9 from . import \_delegators

/usr/local/lib/python3.11/dist-packages/scipy/ndimage/\_ndimage_api.py in <module>
9 from .\_filters import _ # noqa: F403
10 from .\_fourier import _ # noqa: F403
---> 11 from .\_interpolation import _ # noqa: F403
12 from .\_measurements import _ # noqa: F403
13 from .\_morphology import \* # noqa: F403

/usr/local/lib/python3.11/dist-packages/scipy/ndimage/\_interpolation.py in <module>
35 from scipy.\_lib.\_util import normalize_axis_index
36
---> 37 from scipy import special
38 from . import \_ni_support
39 from . import \_nd_image

/usr/lib/python3.11/importlib/_bootstrap.py in \_handle_fromlist(module, fromlist, import_, recursive)

/usr/local/lib/python3.11/dist-packages/scipy/**init**.py in **getattr**(name)
132 def **getattr**(name):
133 if name in submodules:
--> 134 return \_importlib.import_module(f'scipy.{name}')
135 else:
136 try:

/usr/lib/python3.11/importlib/**init**.py in import_module(name, package)
124 break
125 level += 1
--> 126 return \_bootstrap.\_gcd_import(name[level:], package, level)
127
128

/usr/local/lib/python3.11/dist-packages/scipy/special/**init**.py in <module>
824 chdtr, chdtrc, betainc, betaincc, stdtr)
825
--> 826 from . import \_basic
827 from .\_basic import \*
828

/usr/local/lib/python3.11/dist-packages/scipy/special/\_basic.py in <module>
20 from . import \_specfun
21 from .\_comb import \_comb_int
---> 22 from .\_multiufuncs import (assoc_legendre_p_all,
23 legendre_p_all)
24 from scipy.\_lib.deprecation import \_deprecated

/usr/local/lib/python3.11/dist-packages/scipy/special/\_multiufuncs.py in <module>
140
141
--> 142 sph_legendre_p = MultiUFunc(
143 sph_legendre_p,
144 r"""sph_legendre_p(n, m, theta, \*, diff_n=0)

/usr/local/lib/python3.11/dist-packages/scipy/special/\_multiufuncs.py in **init**(self, ufunc_or_ufuncs, doc, force_complex_output, \*\*default_kwargs)
39 for ufunc in ufuncs_iter:
40 if not isinstance(ufunc, np.ufunc):
---> 41 raise ValueError("All ufuncs must have type `numpy.ufunc`."
42 f" Received {ufunc_or_ufuncs}")
43 seen_input_types.add(frozenset(x.split("->")[0] for x in ufunc.types))

ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)
