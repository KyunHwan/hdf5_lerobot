"""HDF5 reading utilities: image decoding, concatenated reads."""

import io

import h5py
import numpy as np
from PIL import Image


def decode_compressed_image(raw_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to HWC uint8 numpy array."""
    img = Image.open(io.BytesIO(raw_bytes))
    return np.array(img)


def get_image_dimensions_from_compressed(raw_bytes: bytes) -> tuple[int, int, int]:
    """Get (H, W, C) from compressed image bytes."""
    img = Image.open(io.BytesIO(raw_bytes))
    w, h = img.size
    c = len(img.getbands())
    return h, w, c


def read_concatenated(h5f: h5py.File, components: list[dict]) -> np.ndarray:
    """Read and concatenate multiple HDF5 datasets along axis=1."""
    arrays = []
    for comp in components:
        data = h5f[comp["hdf5_path"]][:]  # shape [T, D]
        sl = comp.get("slice")
        if sl:
            data = data[:, sl[0]:sl[1]]
        arrays.append(data.astype(np.float32))
    return np.concatenate(arrays, axis=1)
