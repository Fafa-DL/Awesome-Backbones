import io

import cv2
import numpy as np
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)
from utils.misc import is_str


supported_backends = ['cv2', 'turbojpeg', 'pillow', 'tifffile']

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}

imread_backend = 'cv2'


def imfrombytes(content, flag='color', channel_order='bgr', backend=None):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        channel_order (str): The channel order of the output, candidates
            are 'bgr' and 'rgb'. Default to 'bgr'.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`. If backend is
            None, the global imread_backend specified by ``mmcv.use_backend()``
            will be used. Default: None.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> img_path = '/path/to/img.jpg'
        >>> with open(img_path, 'rb') as f:
        >>>     img_buff = f.read()
        >>> img = mmcv.imfrombytes(img_buff)
        >>> img = mmcv.imfrombytes(img_buff, flag='color', channel_order='rgb')
        >>> img = mmcv.imfrombytes(img_buff, backend='pillow')
        >>> img = mmcv.imfrombytes(img_buff, backend='cv2')
    """

    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(
            f'backend: {backend} is not supported. Supported '
            "backends are 'cv2', 'turbojpeg', 'pillow', 'tifffile'")

    img_np = np.frombuffer(content, np.uint8)
    flag = imread_flags[flag] if is_str(flag) else flag
    img = cv2.imdecode(img_np, flag)
    if flag == IMREAD_COLOR and channel_order == 'rgb':
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img