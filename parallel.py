"""Parallelism composition — DeviceMesh for DP x TP x PP x CP.

OLMo-core builds a multi-dimensional DeviceMesh:
    Mesh shape: (pp, dp_replicate, dp_shard, cp, tp)
    Dim names:  ("pp", "dp_replicate", "dp_shard", "cp", "tp")

Sub-meshes:
    get_dp_model_mesh():  all dp* dims flattened (for FSDP/DDP)
    get_dp_mesh():        for data loading (CP ranks get same data)
    get_tp_mesh():        just "tp" dimension
    get_cp_mesh():        just "cp" dimension
    get_pp_mesh():        just "pp" dimension

Flattening rules:
    CP gets flattened into adjacent DP dimension (CP ranks hold full params)
    EP gets flattened into DP dimension (EP ranks hold different experts)

Reference: OLMo-core src/olmo_core/distributed/parallel/__init__.py
           PyTorch torch.distributed.device_mesh
"""

import logging
from typing import List, Optional, Tuple

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

log = logging.getLogger(__name__)


# Mesh dimension names (OLMo-core convention)
class MeshDimName:
    dp = "dp"
    dp_replicate = "dp_replicate"
    dp_shard = "dp_shard"
    tp = "tp"
    cp = "cp"
    pp = "pp"
    ep = "ep"


_WORLD_MESH: Optional[DeviceMesh] = None


def get_world_mesh() -> Optional[DeviceMesh]:
    """Get the global world mesh."""
    return _WORLD_MESH


def build_world_mesh(
    *,
    world_size: Optional[int] = None,
    dp_size: Optional[int] = None,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
    dp_shard: Optional[int] = None,
    dp_replicate: Optional[int] = None,
    device_type: str = "cpu",
) -> DeviceMesh:
    """Build a multi-dimensional DeviceMesh for composable parallelism.

    OLMo-core pattern:
        Mesh dimensions: (pp, dp_replicate, dp_shard, cp, tp)
        Total: pp * dp_replicate * dp_shard * cp * tp = world_size

    :param world_size: total number of GPUs (auto-detected if None).
    :param dp_size: data parallelism degree (auto-computed if None).
    :param tp_size: tensor parallelism degree.
    :param pp_size: pipeline parallelism degree.
    :param cp_size: context parallelism degree.
    :param ep_size: expert parallelism degree.
    :param dp_shard: HSDP shard degree (within-node).
    :param dp_replicate: HSDP replicate degree (across-nodes).
    :param device_type: "cpu" or "cuda".
    :returns: DeviceMesh with named dimensions.
    """
    global _WORLD_MESH

    if _WORLD_MESH is not None:
        raise RuntimeError("World mesh already exists!")

    if world_size is None:
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Auto-compute DP size
    if dp_size is None:
        dp_size = world_size // (tp_size * pp_size * cp_size)
        assert world_size == tp_size * pp_size * cp_size * dp_size, (
            f"world_size ({world_size}) must equal "
            f"tp_size ({tp_size}) * pp_size ({pp_size}) * "
            f"cp_size ({cp_size}) * dp_size ({dp_size})"
        )

    # Build mesh dimensions: (pp, dp, cp, tp)
    mesh_shape = [pp_size]
    dim_names: List[str] = [MeshDimName.pp]

    # HSDP or standard DP
    if dp_shard is not None and dp_replicate is not None:
        mesh_shape.extend([dp_replicate, dp_shard])
        dim_names.extend([MeshDimName.dp_replicate, MeshDimName.dp_shard])
    else:
        mesh_shape.append(dp_size)
        dim_names.append(MeshDimName.dp)

    if cp_size > 1:
        mesh_shape.append(cp_size)
        dim_names.append(MeshDimName.cp)

    if tp_size > 1:
        mesh_shape.append(tp_size)
        dim_names.append(MeshDimName.tp)

    mesh = init_device_mesh(device_type, tuple(mesh_shape), mesh_dim_names=tuple(dim_names))
    log.info(f"Built device mesh: {dim_names} = {mesh_shape}")

    _WORLD_MESH = mesh
    return mesh


def get_dp_model_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Get the sub-mesh for FSDP/DDP (CP flattened into DP).

    CP ranks hold full model params, so gradient sync must happen
    across both DP and CP ranks.

    :param device_mesh: world mesh from build_world_mesh().
    """
    dim_names = device_mesh.mesh_dim_names
    if dim_names is None:
        raise RuntimeError("Mesh has no dimension names")

    # CP gets flattened into adjacent DP dimension
    if MeshDimName.cp in dim_names:
        dp_idx = list(dim_names).index(MeshDimName.cp) - 1
        dp_dim = dim_names[dp_idx]
        assert dp_dim.startswith("dp")
        device_mesh = device_mesh[tuple(n for n in dim_names if n != MeshDimName.cp)]

    dp_dim_names = tuple(n for n in device_mesh.mesh_dim_names if n.startswith("dp"))
    return device_mesh[dp_dim_names]


def get_dp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Get the DP sub-mesh for data loading (CP ranks get same data).

    :param device_mesh: world mesh from build_world_mesh().
    """
    dim_names = device_mesh.mesh_dim_names
    if dim_names is None:
        raise RuntimeError("Mesh has no dimension names")
    dp_dim_names = tuple(n for n in dim_names if n.startswith("dp"))
    return device_mesh[dp_dim_names]


def get_tp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Get the tensor parallel sub-mesh."""
    dim_names = device_mesh.mesh_dim_names
    if dim_names is None or MeshDimName.tp not in dim_names:
        raise RuntimeError("No TP dimension in mesh")
    return device_mesh[MeshDimName.tp]


def get_cp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Get the context parallel sub-mesh."""
    dim_names = device_mesh.mesh_dim_names
    if dim_names is None or MeshDimName.cp not in dim_names:
        raise RuntimeError("No CP dimension in mesh")
    return device_mesh[MeshDimName.cp]


def get_pp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Get the pipeline parallel sub-mesh."""
    dim_names = device_mesh.mesh_dim_names
    if dim_names is None or MeshDimName.pp not in dim_names:
        raise RuntimeError("No PP dimension in mesh")
    return device_mesh[MeshDimName.pp]


def get_ep_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Get the expert parallel sub-mesh."""
    dim_names = device_mesh.mesh_dim_names
    if dim_names is None:
        raise RuntimeError("Mesh has no dimension names")
    if MeshDimName.ep in dim_names:
        return device_mesh[MeshDimName.ep]
    if MeshDimName.dp_shard in dim_names:
        return device_mesh[MeshDimName.dp_shard]
    raise RuntimeError("No EP or DP-shard dimension in mesh")
