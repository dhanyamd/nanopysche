"""Parallelism composition — DeviceMesh for DP x TP x PP x CP x EP.

Two APIs:
  1. build_world_mesh(): manual mesh construction (existing)
  2. ParallelDims: config-driven composable parallelism (TorchTitan pattern)

OLMo-core builds a multi-dimensional DeviceMesh:
    Mesh shape: (pp, dp_replicate, dp_shard, cp, tp)
    Dim names:  ("pp", "dp_replicate", "dp_shard", "cp", "tp")

TorchTitan pattern (ParallelDims):
    ParallelDims(dp_shard=8, tp=8, pp=1, cp=1, ep=1)
    -> auto-validates, computes dp_replicate, builds mesh

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
           TorchTitan torchtitan/distributed.parallel_dims.py
"""

import logging
from dataclasses import dataclass, field
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
        dp_size = world_size // (tp_size * pp_size * cp_size * ep_size)
        assert world_size == tp_size * pp_size * cp_size * dp_size * ep_size, (
            f"world_size ({world_size}) must equal "
            f"tp_size ({tp_size}) * pp_size ({pp_size}) * "
            f"cp_size ({cp_size}) * dp_size ({dp_size}) * "
            f"ep_size ({ep_size})"
        )

    # Build mesh dimensions: (pp, dp, ep, cp, tp)
    mesh_shape = [pp_size]
    dim_names: List[str] = [MeshDimName.pp]

    # HSDP or standard DP
    if dp_shard is not None and dp_replicate is not None:
        mesh_shape.extend([dp_replicate, dp_shard])
        dim_names.extend([MeshDimName.dp_replicate, MeshDimName.dp_shard])
    else:
        mesh_shape.append(dp_size)
        dim_names.append(MeshDimName.dp)

    if ep_size > 1:
        mesh_shape.append(ep_size)
        dim_names.append(MeshDimName.ep)

    if cp_size > 1:
        mesh_shape.append(cp_size)
        dim_names.append(MeshDimName.cp)

    if tp_size > 1:
        mesh_shape.append(tp_size)
        dim_names.append(MeshDimName.tp)

    mesh = init_device_mesh(
        device_type, tuple(mesh_shape), mesh_dim_names=tuple(dim_names)
    )
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


# ---------------------------------------------------------------------------
# ParallelDims — config-driven composable parallelism (TorchTitan pattern)
# ---------------------------------------------------------------------------


@dataclass
class ParallelDims:
    """Config-driven parallelism dimensions.

    Like TorchTitan's ParallelDims, this validates and auto-computes
    derived dimensions from a flat config. Usage:

        dims = ParallelDims(dp_shard=8, tp=8)
        # Validates: 8 * 8 = world_size
        dims.build_mesh(device_type="cuda")
        # -> DeviceMesh with dp_shard=8, tp=8

    HSDP (hybrid sharded data parallel):
        dims = ParallelDims(dp_shard=4, dp_replicate=2, tp=4)
        # Validates: 4 * 2 * 4 = 32 = world_size

    Expert parallelism:
        dims = ParallelDims(dp_shard=8, tp=1, ep=8)
        # Experts sharded across 8 ranks, data replicated across 8

    Reference: TorchTitan torchtitan/distributed/parallel_dims.py
    """

    dp_shard: int = 1
    dp_replicate: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1

    _world_size: Optional[int] = field(default=None, repr=False)
    _mesh: Optional[DeviceMesh] = field(default=None, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate that dimensions are consistent and compute derived values."""
        if self.dp_shard < 1:
            raise ValueError(f"dp_shard must be >= 1, got {self.dp_shard}")
        if self.tp < 1:
            raise ValueError(f"tp must be >= 1, got {self.tp}")
        if self.pp < 1:
            raise ValueError(f"pp must be >= 1, got {self.pp}")
        if self.cp < 1:
            raise ValueError(f"cp must be >= 1, got {self.cp}")
        if self.ep < 1:
            raise ValueError(f"ep must be >= 1, got {self.ep}")
        if self.dp_replicate < 1:
            raise ValueError(f"dp_replicate must be >= 1, got {self.dp_replicate}")

    @property
    def world_size(self) -> int:
        return self.dp_shard * self.dp_replicate * self.tp * self.pp * self.cp * self.ep

    def set_world_size(self, world_size: int):
        """Set world size and validate. Auto-computes dp_replicate if needed."""
        self._world_size = world_size
        if self.dp_replicate == 1:
            # Auto-compute dp_replicate from remaining dims
            remainder = world_size // (
                self.dp_shard * self.tp * self.pp * self.cp * self.ep
            )
            if (
                remainder * self.dp_shard * self.tp * self.pp * self.cp * self.ep
                != world_size
            ):
                raise ValueError(
                    f"world_size ({world_size}) is not divisible by "
                    f"dp_shard({self.dp_shard}) * tp({self.tp}) * pp({self.pp}) * "
                    f"cp({self.cp}) * ep({self.ep})"
                )
            self.dp_replicate = remainder
        else:
            if self.world_size != world_size:
                raise ValueError(
                    f"Parallel dims ({self.world_size}) != world_size ({world_size})"
                )

    def build_mesh(self, device_type: str = "cuda") -> DeviceMesh:
        """Build a multi-dimensional DeviceMesh from these dims.

        :param device_type: "cpu" or "cuda".
        :returns: DeviceMesh with named dimensions.
        """
        global _WORLD_MESH

        if self._mesh is not None:
            return self._mesh

        # Build mesh dimensions: (pp, dp_replicate, dp_shard, ep, cp, tp)
        # Order matters for DTensor placement
        mesh_shape = [self.pp]
        dim_names: List[str] = [MeshDimName.pp]

        if self.dp_replicate > 1:
            mesh_shape.append(self.dp_replicate)
            dim_names.append(MeshDimName.dp_replicate)

        mesh_shape.append(self.dp_shard)
        dim_names.append(MeshDimName.dp_shard)

        if self.ep > 1:
            mesh_shape.append(self.ep)
            dim_names.append(MeshDimName.ep)

        if self.cp > 1:
            mesh_shape.append(self.cp)
            dim_names.append(MeshDimName.cp)

        if self.tp > 1:
            mesh_shape.append(self.tp)
            dim_names.append(MeshDimName.tp)

        mesh = init_device_mesh(
            device_type, tuple(mesh_shape), mesh_dim_names=tuple(dim_names)
        )
        log.info(f"Built device mesh: {dim_names} = {mesh_shape}")

        self._mesh = mesh
        _WORLD_MESH = mesh
        return mesh

    @property
    def mesh(self) -> Optional[DeviceMesh]:
        return self._mesh

    def __repr__(self) -> str:
        return (
            f"ParallelDims(dp_shard={self.dp_shard}, dp_replicate={self.dp_replicate}, "
            f"tp={self.tp}, pp={self.pp}, cp={self.cp}, ep={self.ep})"
        )
