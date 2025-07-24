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
OP_ARGMIN = 12
OP_ARGMAX = 13
OP_ARGMIN_SCALED = 14
OP_ARGMAX_SCALED = 15


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
    elif op_code == OP_ARGMIN:
        op_str = 'argmin'
    elif op_code == OP_ARGMAX:
        op_str = 'argmax'
    elif op_code == OP_ARGMIN_SCALED:
        op_str = 'argmin-scaled'
    elif op_code == OP_ARGMAX_SCALED:
        op_str = 'argmax-scaled'

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
    elif op_str == 'argmin':
        op_code = OP_ARGMIN
    elif op_str == 'argmax':
        op_code = OP_ARGMAX
    elif op_str == 'argmin-scaled':
        op_code = OP_ARGMIN_SCALED
    elif op_str == 'argmax-scaled':
        op_code = OP_ARGMAX_SCALED

    return op_code


def read_maskfiles(expnums, maskfiles, exp_field):
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

        # Check for zero-size regions...
        if 'radius' in subtable.dtype.names:
            gd, = np.where(subtable['radius'] > 0.0)
            if gd.size < subtable.size:
                print("Removing %d regions of zero radius." % (subtable.size - gd.size))
                subtable = subtable[gd]
        elif 'ra_1' in subtable.dtype.names:
            ras = np.vstack((subtable['ra_1'], subtable['ra_2'],
                             subtable['ra_3'], subtable['ra_4']))
            decs = np.vstack((subtable['dec_1'], subtable['dec_2'],
                              subtable['dec_3'], subtable['dec_4']))
            delta_ra = ras.max(axis=0) - ras.min(axis=0)
            delta_dec = decs.max(axis=0) - decs.min(axis=0)

            gd, = np.where((delta_ra > 0.0) & (delta_dec > 0.0))
            if gd.size < subtable.size:
                print("Removing %d regions of zero extent." % (subtable.size - gd.size))
                subtable = subtable[gd]

        a, b = esutil.numpy_util.match(expnums, subtable[exp_field])
        if masktable is None:
            masktable = subtable[b]
        else:
            masktable = np.append(masktable, subtable[b])

    return masktable


def compute_visit_iqr_and_optics_scale(config, table):
    """Compute the visit IQR and optics scale.

    Parameters
    ----------
    config : `decasu.Configuration`
    table : `np.ndarray` or `astropy.table.Table`
    """
    u, inv = np.unique(table[config.exp_field], return_inverse=True)
    h, rev = esutil.stat.histogram(inv, rev=True)
    inds, = np.where(h > 0)

    table[f"{config.fwhm_field}_iqr"][:] = 0.0
    table[f"{config.fwhm_field}_optics_scale"][:] = 1.0

    for ind in inds:
        i1a = rev[rev[ind]: rev[ind + 1]]
        isfinite = np.isfinite(table[config.fwhm_field][i1a]) & (table[config.fwhm_field][i1a] > 0.0)
        if np.any(isfinite):
            five, lo, hi = np.percentile(table[config.fwhm_field][i1a[isfinite]], [5.0, 25.0, 75.0])
            table[f"{config.fwhm_field}_iqr"][i1a[isfinite]] = hi - lo

            table[f"{config.fwhm_field}_optics_scale"][i1a] = table[f"{config.fwhm_field}"][i1a] / five
