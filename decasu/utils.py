
OP_NONE = 0
OP_SUM = 1
OP_MEAN = 2
OP_WMEAN = 3
OP_MIN = 4
OP_MAX = 5
OP_OR = 6


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

    return op_code
