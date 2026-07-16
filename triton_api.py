"""Client-facing interface to the triton_ptx backend.
"""

from triton_ptx.api import (
    dump_kernel_ptx as _dump_kernel_ptx,
    evaluate_candidate as _evaluate_candidate,
    get_kernel_data as _get_kernel_data,
    is_gpu_available as _is_gpu_available,
    list_kernels as _list_kernels,
)


class KernelEval:
    """Attribute-access view over a serializable evaluation result.

    Nested values (payload, reports) are returned as plain dicts, so JSON
    serialization and truthiness match the underlying data.
    """

    __slots__ = ("_data",)

    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self):
        return self._data


def list_kernels():
    return _list_kernels()


def is_gpu_available():
    return _is_gpu_available()


def dump_kernel_ptx(kernel_id):
    return _dump_kernel_ptx(kernel_id)


def get_kernel_data(kernel_id):
    return _get_kernel_data(kernel_id)


def evaluate_candidate(kernel_id, payload):
    return KernelEval(_evaluate_candidate(kernel_id, payload))
