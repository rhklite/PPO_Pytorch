"""
File Description: custom print utilities for debug
Project: python
Author: Daniel Dworakowski, Han Hu
Date: Nov-18-2019
"""
import gc
import inspect
import time
import functools

try:
    import torch
except ModuleNotFoundError:
    print(
        "Package torch was not found."
        "Pytorch tensor related print functions will not be available."
    )

# Color terminal (https://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python).


class Colours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#
# Error information.


def lineInfo():
    callerframerecord = inspect.stack()[2]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    file = info.filename
    file = file[file.rfind('/') + 1 :]
    return '%s::%s:%d' % (file, info.function, info.lineno)


#
# Line information.


def getLineInfo(leveloffset=0):
    level = 2 + leveloffset
    callerframerecord = inspect.stack()[level]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    file = info.filename
    file = file[file.rfind('/') + 1 :]
    return '%s: %d' % (file, info.lineno)


#
# Colours a string.


def colourString(msg, ctype):
    return ctype + msg + Colours.ENDC


#
# Print something in color.


def printColour(msg, ctype=Colours.OKGREEN):
    print(colourString(msg, ctype))


#
# Print information.


def printInfo(*umsg):
    msg = '%s:  ' % (lineInfo())
    lst = ''
    for mstr in umsg:
        if isinstance(mstr, torch.Tensor):
            vname = varname(mstr)
            lst += '[' + str(vname) + ']\n'
        elif not isinstance(mstr, str):
            vname = varname(mstr)
            lst += '[' + str(vname) + ' ' + str(type(mstr)) + '] '
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.OKGREEN) + lst
    print(msg)


#
# Print error information.


def printFrame():
    print(lineInfo(), Colours.WARNING)


#
# Print an error.


def printError(*errstr):
    msg = '%s:  ' % (lineInfo())
    lst = ''
    for mstr in errstr:
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.FAIL) + lst
    print(msg)


#
# Print a warning.


def printWarn(*warnstr):
    msg = '%s:  ' % (lineInfo())
    lst = ''
    for mstr in warnstr:
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.WARNING) + lst
    print(msg)


#
# Get name of variable passed to the function


def varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [
            var_name
            for var_name, var_val in fi.frame.f_locals.items()
            if var_val is var
        ]
        if len(names) > 0:
            return names[0]


def printList(is_different, dlist):
    ret = ''
    if is_different:
        ret = dlist
    else:
        ret = [str(dlist[0])]
    return ret


#
#


def getDevice(t):
    ret = None
    if isinstance(t, torch.Tensor):
        ret = t.device
    else:
        ret = type(t)
    return ret


#
# Get the s


def tensorListInfo(tensor_list, vname, usrmsg, leveloffset):
    assert isinstance(tensor_list, list) or isinstance(tensor_list, tuple)
    str_ret = ''
    dtypes = [tensor_list[0].dtype]
    devices = [tensor_list[0].device]
    shapes = [tensor_list[0].shape]
    dtype_different = False
    devices_different = False
    shapes_different = False
    for t_idx in range(1, len(tensor_list)):
        t = tensor_list[t_idx]
        dtypes.append(t.dtype)
        devices.append(getDevice(t))
        shapes.append(t.shape)
        dtype_different |= t.dtype != dtypes[0]
        devices_different |= t.device != devices[0]
        shapes_different |= t.shape != shapes[0]
    dtypes = printList(dtype_different or devices_different, dtypes)
    devices = printList(dtype_different or devices_different, devices)
    shapes = str(printList(shapes_different, shapes))
    devices_dtypes = ' '.join(map(str, *zip(dtypes, devices)))
    msg = (
        colourString(
            colourString(getLineInfo(leveloffset + 1), Colours.UNDERLINE),
            Colours.OKBLUE,
        )
        + ': ['
        + str(vname)
        + '] '
        + ('<list>' if isinstance(tensor_list, list) else '<tuple>')
        + ' len: %d' % len(tensor_list)
        + ' ('
        + colourString(devices_dtypes, Colours.WARNING)
        + ') -- '
        + colourString('%s' % shapes, Colours.OKGREEN)
        + (' </list>' if isinstance(tensor_list, list) else ' </tuple>')
        + usrmsg
    )
    return msg


#
# Print information about a tensor.


def printTensor(tensor, usrmsg='', leveloffset=0):
    vname = varname(tensor)
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        msg = tensorListInfo(tensor, vname, usrmsg, leveloffset)
    elif isinstance(tensor, torch.Tensor):
        msg = (
            colourString(
                colourString(getLineInfo(leveloffset), Colours.UNDERLINE),
                Colours.OKBLUE,
            )
            + ': ['
            + str(vname)
            + '] ('
            + colourString(
                str(tensor.dtype) + ' ' + str(tensor.device), Colours.WARNING
            )
            + ') -- '
            + colourString('%s' % str(tensor.shape), Colours.OKGREEN)
            + ' '
            + colourString('%s' % str(tensor.grad_fn), Colours.OKGREEN)
            + ' '
            + usrmsg
        )
    else:
        msg = (
            colourString(
                colourString(getLineInfo(leveloffset), Colours.UNDERLINE),
                Colours.OKBLUE,
            )
            + ': ['
            + str(vname)
            + '] ('
            + colourString(
                str(tensor.dtype) + ' ' + str(getDevice(tensor)), Colours.WARNING
            )
            + ') -- '
            + colourString('%s' % str(tensor.shape), Colours.OKGREEN)
            + ' '
            + usrmsg
        )
    print(msg)


#
# Print debugging information.


def dprint(usrmsg, leveloffset=0):
    msg = (
        colourString(
            colourString(getLineInfo(leveloffset), Colours.UNDERLINE), Colours.OKBLUE
        )
        + ': '
        + str(usrmsg)
    )
    print(msg)


def hasNAN(t):
    msg = (
        colourString(colourString(getLineInfo(), Colours.UNDERLINE), Colours.OKBLUE)
        + ': '
        + colourString(
            str('Tensor has %s NaNs' % str((t != t).sum().item())), Colours.FAIL
        )
    )
    print(msg)


def torch_mem():
    dprint(
        'Torch report: Allocated: %.2f MBytes Cached: %.2f'
        % (
            torch.cuda.memory_allocated() / (1024 ** 2),
            torch.cuda.memory_cached() / (1024 ** 2),
        ),
        1,
    )


# MEM utils


def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported
    https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation'''
        print('Storage on %s' % (mem_type))
        print('-' * LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for idx, tensor in enumerate(tensors):
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)
            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())
            print(
                '{:3} {}\t\t{}\t\t{:.2f}\t\t{}'.format(
                    idx, element_type, size, mem, tensor.grad_fn
                )
            )
        print('-' * LEN)
        print(
            'Total Tensors: %d \tUsed Memory Space: %.5f MBytes'
            % (total_numel, total_mem)
        )
        print(
            'Torch report: %.2f MBytes' % (torch.cuda.memory_allocated() / (1024 ** 2))
        )
        print('-' * LEN)

    LEN = 65
    print('=' * LEN)
    gc.collect()
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' % ('Element type', 'Size', 'Used MEM(MBytes)'))
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('=' * LEN)


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()  # 1
        value = func(*args, **kwargs)
        end_time = time.time()  # 2
        run_time = end_time - start_time  # 3
        printColour("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value

    return wrapper_timer


def debug(func):
    """Print the function signature and return value"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]  # 1
        kwargs_repr = ["{}={}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)  # 3
        printColour("Calling {}({})".format(func.__name__, signature))
        value = func(*args, **kwargs)
        printColour("{} returned {}".format(func.__name__, value))  # 4
        return value

    return wrapper_debug
