from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_vectors(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    if vectors.ndim != 2 or len(ids) != vectors.shape[0]:
        raise ValueError(f"ids ({len(ids)}) and vectors {vectors.shape} do not line up")
    flat = pa.array(np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1))
    column = pa.FixedSizeListArray.from_arrays(flat, vectors.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so an interrupted write never destroys the previous file.
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.table({"id": pa.array(ids, pa.string()), "vector": column}), tmp)
    tmp.replace(path)


def read_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    table = pq.read_table(path)
    ids = table.column("id").to_pylist()
    column = table.column("vector").combine_chunks()
    dim = column.type.list_size
    flat = column.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return ids, flat.reshape(-1, dim)
