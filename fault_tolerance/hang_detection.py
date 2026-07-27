from __future__ import annotations

"""Hang detection — detect and kill stuck ranks in distributed training.

In multi-node training, ranks can hang due to:
    1. NCCL timeout (network issues)
    2. GPU errors (ECC, thermal)
    3. Deadlock (collective mismatch between ranks)
    4. Infinite loops (code bugs)

Without hang detection, one stuck rank blocks ALL ranks (since collectives
are blocking). The cluster sits idle burning compute credits.

Detection patterns:
    1. Timeout-based: if a rank hasn't progressed in N seconds, it's hung
    2. Heartbeat: ranks periodically signal liveness, missing heartbeat = hang
    3. Watchdog: separate thread monitors main thread progress
    4. NCCL timeout: dist.init_process_group(timeout=...)

OLMo-core:
    - Uses PyTorch's built-in NCCL timeout
    - Checkpointing allows resume after killing stuck ranks

DisTrO:
    - Designed for unreliable hardware with aggressive timeout detection
    - Kill-and-resume pattern for consumer GPU clusters

Reference: PyTorch distributed timeout docs
"""

import torch
import torch.distributed as dist
import time
import threading
from typing import Callable


class HangDetector:
    """Monitor distributed training for hung ranks.

    How it works:
        1. Main thread updates a "last progress" timestamp
        2. Watchdog thread periodically checks if any rank is stale
        3. If stale beyond timeout, triggers recovery action

    Recovery actions:
        - Log the hang (for debugging)
        - Kill the process group (abort all ranks)
        - Checkpoint before killing (if possible)
        - Signal torchelastic to restart

    Usage:
        detector = HangDetector(timeout=300, on_hang=recovery_fn)
        detector.start()
        for step in range(max_steps):
            train_step()
            detector.heartbeat()
        detector.stop()
    """

    def __init__(
        self,
        timeout: float = 300.0,
        check_interval: float = 10.0,
        on_hang: Callable | None = None,
    ):
        self.timeout = timeout
        self.check_interval = check_interval
        self.on_hang = on_hang
        self._last_heartbeat = time.time()
        self._running = False
        self._thread = None

    def start(self):
        """Start the watchdog thread."""
        self._running = True
        self._last_heartbeat = time.time()
        self._thread = threading.Thread(target=self._watchdog, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the watchdog thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def heartbeat(self):
        """Signal that this rank is alive. Call after each training step."""
        self._last_heartbeat = time.time()

    def _watchdog(self):
        """Background thread that monitors for hangs."""
        while self._running:
            time.sleep(self.check_interval)

            elapsed = time.time() - self._last_heartbeat
            if elapsed > self.timeout:
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(
                    f"[HangDetector] Rank {rank} detected hang: "
                    f"{elapsed:.1f}s since last heartbeat (timeout={self.timeout}s)"
                )

                if self.on_hang is not None:
                    self.on_hang()

                # Abort the process group
                if dist.is_initialized():
                    dist.abort()

                break

    def check_collective_timeout(
        self,
        collective_fn: Callable,
        timeout: float = 60.0,
    ) -> bool:
        """Try a collective with a timeout. Returns True if it hangs.

        Useful for detecting deadlocked collectives (e.g., one rank
        called all_reduce but another didn't).
        """
        try:
            collective_fn()
            return False
        except Exception as e:
            if "timeout" in str(e).lower():
                return True
            raise


class HeartbeatCoordinator:
    """Coordinate heartbeats across all ranks via all-reduce barrier.

    Instead of point-to-point heartbeats (which adds complexity), this uses
    a periodic all-reduce barrier: all ranks participate in a lightweight
    all-reduce every `check_interval` seconds. If a rank misses the barrier,
    its peers detect the hang via timeout.

    This is OLMo-core's pattern — simple, robust, no P2P state to manage.

    Each rank:
      1. Increments a local step counter
      2. Periodically (every check_interval) does an all-reduce of the counter
      3. Rank 0 verifies all ranks' counters advanced since last check

    :param check_interval: seconds between heartbeat checks.
    :param hang_timeout: seconds without progress before declaring hang.
    :param group: process group for heartbeat collectives.
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        hang_timeout: float = 120.0,
        group: dist.ProcessGroup | None = None,
    ):
        self.check_interval = check_interval
        self.hang_timeout = hang_timeout
        self.group = group or dist.group.WORLD
        self._step_counter = 0
        self._last_check_counter = {
            i: -1 for i in range(dist.get_world_size(self.group))
        }
        self._running = False
        self._thread: threading.Thread | None = None

    def heartbeat(self):
        """Call after each training step to increment local counter."""
        self._step_counter += 1

    def start(self):
        """Start coordinator thread on rank 0."""
        rank = dist.get_rank(self.group)
        self._running = True
        if rank == 0:
            self._last_check_counter = {
                i: -1 for i in range(dist.get_world_size(self.group))
            }
            self._thread = threading.Thread(target=self._coordinator_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop coordinator."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def sync_heartbeats(self):
        """All ranks: broadcast step counters via all-gather."""
        world = dist.get_world_size(self.group)
        rank = dist.get_rank(self.group)
        local_tensor = torch.tensor([self._step_counter], dtype=torch.long)
        gathered = [torch.zeros(1, dtype=torch.long) for _ in range(world)]
        dist.all_gather(gathered, local_tensor, group=self.group)
        return {r: gathered[r].item() for r in range(world)}

    def _coordinator_loop(self):
        """Rank 0: periodically gather heartbeats and detect hangs."""
        while self._running:
            time.sleep(self.check_interval)

            try:
                counters = self.sync_heartbeats()
            except RuntimeError:
                continue

            for r, c in counters.items():
                if c == self._last_check_counter[r]:
                    elapsed_hint = self.check_interval * 2
                    if elapsed_hint >= self.hang_timeout:
                        print(
                            f"[HeartbeatCoordinator] Rank {r} appears hung: "
                            f"counter stuck at {c} for ~{elapsed_hint:.0f}s"
                        )
                        if dist.is_initialized():
                            dist.destroy_process_group()
                        break
            self._last_check_counter = counters
