#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np

from cuml.common.sparse_utils import is_sparse
from cuml.internals.validation import check_array

_is_axis_1_sorted_kernel = cp.RawKernel(
    """
    extern "C" __global__
    void is_axis_1_sorted(const float* arr, int n_rows, int n_cols, int *sorted) {
        int row = blockDim.x * blockIdx.x + threadIdx.x;

        if(row >= n_rows) return;

        int start = row * n_cols;
        int end = start + n_cols - 1;
        for(int i = start; i < end; i++) {
            if (arr[i] > arr[i + 1]) {
                *sorted = 0;
                return;
            }
        }
    }
    """,
    "is_axis_1_sorted",
)


def _is_axis_1_sorted(X):
    """Checks if axis 1 (every row) is sorted in X."""
    # Safety checks
    assert X.flags.c_contiguous
    assert X.dtype == "float32"
    assert X.ndim == 2
    # XXX: ensure on device just for this routine - zero copy unless a
    # host->device transfer needed.
    X = cp.asarray(X, order="C")
    is_sorted = cp.ones(1, dtype="int32")
    _is_axis_1_sorted_kernel(
        (math.ceil(X.shape[0] / 32),),
        (32,),
        (X, X.shape[0], X.shape[1], is_sorted),
    )
    return bool(is_sorted.item())


def extract_knn_graph(
    knn_info,
    n_neighbors,
    mem_type="device",
    indices_dtype="int64",
):
    """Extract the KNN graph indices and distances.

    Parameters
    ----------
    knn_info : array, sparse-matrix, or tuple[array, array]
        - Pairwise distances dense array of shape (n_samples, n_samples)
        - KNN graph sparse array (preferably CSR)
        - Tuple (indices, distances) of arrays of shape (n_samples, n_neighbors)
    n_neighbors: int
        Number of nearest neighbors
    mem_type : {"device", "host", None}, default="device"
        The desired output memory type.
    indices_dtype : dtype, default='int64'
        The dtype to use for the output indices.

    Returns
    -------
    indices : cupy.ndarray or numpy.ndarray
        The KNN indices, shape=n_samples * n_neighbors, dtype=indices_dtype.
    distances : cupy.ndarray or numpy.ndarray
        The KNN distances, shape=n_samples * n_neighbors, dtype=float32.
    """
    # The initial mem_type to coerce to. When possible we only coerce to device
    # if the output is known to be device, otherwise we leave as is until the
    # final coercion.
    mem_type_init = "device" if mem_type == "device" else None
    if isinstance(knn_info, tuple):
        # (indices, distances), each with shape=(n_samples, orig_n_neighbors)
        indices, distances = knn_info
        indices = check_array(
            indices, dtype=indices_dtype, order="C", mem_type=mem_type_init
        )
        if mem_type is None:
            mem_type = "device" if isinstance(indices, cp.ndarray) else "host"
        distances = check_array(
            distances, dtype="float32", order="C", mem_type=mem_type_init
        )
        if not indices.shape == distances.shape:
            raise ValueError(
                f"Expected indices and distances to have shape=(n_samples, "
                f"n_neighbors), got indices.shape={indices.shape}, "
                f"distances.shape={distances.shape}"
            )
    elif is_sparse(knn_info):
        # Sparse KNN graph
        # - shape=(n_samples, n_samples)
        # - nnz=n_samples * orig_n_neighbors
        if mem_type is None:
            mem_type = "device" if cp_sp.issparse(knn_info) else "host"
        if not (knn_info.ndim == 2 and knn_info.shape[0] == knn_info.shape[1]):
            raise ValueError(
                f"Expected a sparse array of shape=(n_samples, n_samples), "
                f"got shape={knn_info.shape}"
            )
        n_samples = knn_info.shape[0]
        if knn_info.nnz % n_samples != 0:
            raise ValueError(
                f"Precomputed KNN graph has {knn_info.nnz} total elements which "
                f"is not evenly divisible by {n_samples} samples."
            )
        orig_n_neighbors = knn_info.nnz // n_samples

        # Coerce to CSR. If the input was already CSR this is zero-copy and
        # avoids reordering indices (leaving `.data` in the initial order).
        # This ensures the case of passing a direct `kneighbors_graph` output
        # can be done zero-copy.
        knn_info = knn_info.tocsr()
        indices = check_array(
            knn_info.indices.reshape((n_samples, orig_n_neighbors)),
            dtype=indices_dtype,
            order="C",
            mem_type=mem_type_init,
        )
        distances = check_array(
            knn_info.data.reshape((n_samples, orig_n_neighbors)),
            dtype="float32",
            order="C",
            mem_type=mem_type_init,
        )
        # Reorder by distances if not already sorted. This is necessary for KNN
        # graph inputs, since a canonical sparse matrix will not have the data
        # sorted as we need it. Both sklearn and cuml's `kneighbors_graph`
        # returns a matrix with `.data` sorted as required (not canonicalized),
        # so we optimistically check for sortedness before doing the sorting.
        if not _is_axis_1_sorted(distances):
            xp = cp if isinstance(distances, cp.ndarray) else np
            new_order = distances.argsort()
            all_rows = xp.arange(distances.shape[0])[:, None]
            indices = indices[all_rows, new_order]
            distances = distances[all_rows, new_order]
            del new_order
    else:
        # Dense pairwise distance matrix, shape=(n_samples, n_samples)

        # XXX: We always convert pairwise matrices to distances and indices on
        # device. This requires mutation of the input array - to ensure only a
        # single copy is made while handling `mem_type` requires some care.

        # - Validate, ensuring a copy if requested mem_type is device
        knn_info = check_array(
            knn_info,
            dtype="float32",
            mem_type=mem_type_init,
            copy=mem_type_init == "device",
        )
        # - Coerce `mem_type=None` to a concrete mem_type
        if mem_type is None:
            mem_type = "device" if isinstance(knn_info, cp.ndarray) else "host"
        # - Ensure a copy and device output if this wasn't already done above
        if mem_type_init != "device":
            knn_info = cp.asarray(knn_info, copy=True)

        if knn_info.shape[0] != knn_info.shape[1]:
            raise ValueError(
                f"Expected a dense array of shape=(n_samples, n_samples), "
                f"got shape={knn_info.shape}"
            )
        n_samples = knn_info.shape[0]
        if n_samples < n_neighbors:
            raise ValueError(
                f"Precomputed KNN data requires n_samples >= n_neighbors. "
                f"Got {n_neighbors=}, {n_samples=}"
            )

        # Convert pairwise distance matrix to KNN graph
        # Fill diagonal with inf so diagonal indices sort last
        cp.fill_diagonal(knn_info, cp.inf)
        # Partition indices to select the nearest `n_neighbors`
        indices = cp.argpartition(knn_info, n_neighbors - 1, axis=1)
        indices = indices[:, :n_neighbors]
        # Reorder and subset indices and distances appropriately
        all_rows = cp.arange(n_samples)[:, None]
        indices = indices[all_rows, cp.argsort(knn_info[all_rows, indices])]
        distances = knn_info[all_rows, indices]

    # Validate shape and n_neighbors
    if indices.shape[1] < n_neighbors:
        raise ValueError(
            f"Precomputed KNN data has {indices.shape[1]} neighbors per "
            f"sample, but {n_neighbors=} was specified. Please provide KNN data "
            f"with at least {n_neighbors} neighbors per sample."
        )

    # Trim arrays to n_neighbors if necessary
    if indices.shape[1] > n_neighbors:
        indices = indices[:, :n_neighbors]
        distances = distances[:, :n_neighbors]

    # Reshape and coerce to proper dtype and mem_type.
    indices = check_array(
        indices.reshape(-1),
        dtype=indices_dtype,
        mem_type=mem_type,
        ensure_2d=False,
    )
    distances = check_array(
        distances.reshape(-1),
        dtype="float32",
        mem_type=mem_type,
        ensure_2d=False,
        ensure_all_finite=False,
    )
    return indices, distances
