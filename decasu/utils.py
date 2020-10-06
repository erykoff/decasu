import esutil
import fitsio
import numpy as np


OP_NONE = 0
OP_SUM = 1
OP_MEAN = 2
OP_WMEAN = 3
OP_MIN = 4
OP_MAX = 5
OP_OR = 6
OP_SUM_SCALED = 7
OP_MEAN_SCALED = 8
OP_WMEAN_SCALED = 9
OP_MIN_SCALED = 10
OP_MAX_SCALED = 11


valid_map_types_basic = ['nexp']


def op_code_to_str(op_code):
    """
    Convert supreme op_code to string

    Parameters
    ----------
    op_code : `int`
       Operation code number

    Returns
    -------
    op_str : `str`
       String representation of op_code
    """
    if op_code == OP_SUM:
        op_str = 'sum'
    elif op_code == OP_MEAN:
        op_str = 'mean'
    elif op_code == OP_WMEAN:
        op_str = 'wmean'
    elif op_code == OP_MIN:
        op_str = 'min'
    elif op_code == OP_MAX:
        op_str = 'max'
    elif op_code == OP_OR:
        op_str = 'or'
    elif op_code == OP_SUM_SCALED:
        op_str = 'sum-scaled'
    elif op_code == OP_MEAN_SCALED:
        op_str = 'mean-scaled'
    elif op_code == OP_WMEAN_SCALED:
        op_str = 'wmean-scaled'
    elif op_code == OP_MIN_SCALED:
        op_str = 'min-scaled'
    elif op_code == OP_MAX_SCALED:
        op_str = 'max-scaled'

    return op_str


def op_str_to_code(op_str):
    """
    Convert supreme operation string to code

    Parameters
    ----------
    op_str : `str`
       String representation of op_code

    Returns
    -------
    op_code : `int`
       Operation code number
    """
    if op_str == 'sum':
        op_code = OP_SUM
    elif op_str == 'mean':
        op_code = OP_MEAN
    elif op_str == 'wmean':
        op_code = OP_WMEAN
    elif op_str == 'min':
        op_code = OP_MIN
    elif op_str == 'max':
        op_code = OP_MAX
    elif op_str == 'or':
        op_code = OP_OR
    elif op_str == 'sum-scaled':
        op_code = OP_SUM_SCALED
    elif op_str == 'mean-scaled':
        op_code = OP_MEAN_SCALED
    elif op_str == 'wmean-scaled':
        op_code = OP_WMEAN_SCALED
    elif op_str == 'min-scaled':
        op_code = OP_MIN_SCALED
    elif op_str == 'max-scaled':
        op_code = OP_MAX_SCALED

    return op_code


def read_maskfiles(expnums, maskfiles):
    """
    Read a list of mask files, cutting to the specific expnums.

    Parameters
    ----------
    expnums : `np.ndarray`
       Exposure numbers to keep
    maskfiles : `list` [`str`]
       Mask files to read

    Returns
    -------
    masktable : `np.ndarray`
       Mask table
    """
    masktable = None
    for maskfile in maskfiles:
        subtable = fitsio.read(maskfile, ext=1, lower=True, trim_strings=True)
        a, b = esutil.numpy_util.match(expnums, subtable['expnum'])
        if masktable is None:
            masktable = subtable[b]
        else:
            masktable = np.append(masktable, subtable[b])

    return masktable
