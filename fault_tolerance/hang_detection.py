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
    """Coordinate heartbeats across all ranks.

    Each rank periodically sends its heartbeat to rank 0.
    Rank 0 monitors all heartbeats and detects hangs.

    This is more robust than per-rank detection because it provides
    a global view of the training state.
    """

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._heartbeats = {}
        self._running = False

    def coordinator_loop(self):
        """Run on rank 0: collect heartbeats and detect hangs."""
        if not dist.is_initialized() or dist.get_rank() != 0:
            return

        self._running = True
        while self._running:
            time.sleep(self.check_interval)

            # Check for stale ranks
            current_time = time.time()
            for rank, last_time in self._heartbeats.items():
                if current_time - last_time > self.check_interval * 3:
                    print(
                        f"[HeartbeatCoordinator] Rank {rank} appears hung: "
                        f"{current_time - last_time:.1f}s since last heartbeat"
                    )
