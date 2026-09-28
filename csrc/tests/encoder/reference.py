"""Independent NumPy references for the supplied encoder's custom activations.

Line numbers refer to LF-delimited cuda_dldn_swin_enc0_kernel.ptx.txt.
These functions accept scalars or arrays and preserve all half rounding points.
"""

import numpy as np


def encoded_exponential(score):
    """PTX LF 12432-12504: scaled/clipped polynomial, then half bit encoding.

    The scale belongs to this operator. Do not apply an additional QK scale.
    The two half FMAs use float64 to evaluate the exact half-input polynomial
    before the single rounding to float16 at each FMA result.
    """
    value = np.asarray(score, dtype=np.float16)
    scale = np.float16(np.float32(0.017186790704727173))
    bound = np.float16(0.55615234375)
    coefficient = np.float16(0.92724609375)
    offset = np.float16(1.375)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.asarray(value * scale, dtype=np.float16)
        clipped = np.fmin(np.fmax(scaled, -bound), bound)
        wide = np.asarray(clipped, dtype=np.float64)
        quadratic = np.asarray(
            wide * (-wide) + np.float64(coefficient), dtype=np.float16
        )
        encoded = np.asarray(
            wide * np.asarray(quadratic, dtype=np.float64) + np.float64(offset),
            dtype=np.float16,
        )
    bits = np.asarray((encoded.view(np.uint16).astype(np.uint32) << 5) & 0x7FE0,
                      dtype=np.uint16)
    result = bits.view(np.float16)
    return result[()] if result.ndim == 0 else result


def clipped_gelu(x):
    """PTX LF 40947-41008: custom clipped gate, not erf/tanh GELU.

    The five arithmetic operations each round separately to float16.
    """
    value = np.asarray(x, dtype=np.float16)
    slope = np.float16(np.float32(0.4121621549129486))
    curvature = np.float16(np.float32(0.0810810774564743))
    clipped = np.fmin(np.fmax(value, np.float16(-2)), np.float16(2))
    with np.errstate(over="ignore", invalid="ignore"):
        magnitude_term = np.asarray(curvature * np.abs(clipped), dtype=np.float16)
        linear_term = np.asarray(slope - magnitude_term, dtype=np.float16)
        signed_term = np.asarray(clipped * linear_term, dtype=np.float16)
        gate = np.asarray(np.float16(0.5) + signed_term, dtype=np.float16)
        result = np.asarray(value * gate, dtype=np.float16)
    return result[()] if result.ndim == 0 else result
