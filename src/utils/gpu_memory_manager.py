"""
GPU Memory Management Utilities for SGLang LoRA MoE Debug Agent.

This module provides utilities to maintain persistent GPU memory reservation
throughout the framework's lifetime, with temporary release during test execution.

Strategy:
1. Reserve GPU memory when framework starts
2. Release reservation immediately before test execution
3. Re-reserve memory immediately after test completion
4. Maintain reservation during all other framework operations

This prevents other processes on shared nodes from claiming GPU memory
during framework idle time.
"""

import atexit
import gc
from typing import Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class GPUMemoryManager:
    """
    Manages persistent GPU memory reservation for shared node environments.

    This class maintains GPU memory reservation throughout the framework's lifetime,
    temporarily releasing it only during test execution to prevent other processes
    from claiming memory on shared compute nodes.
    """

    def __init__(self, device: int = 0, reserve_gb: float = 2.0, persistent: bool = True):
        """
        Initialize GPU memory manager.

        Args:
            device: CUDA device ID (default: 0)
            reserve_gb: Amount of GPU memory to reserve in GB (default: 2.0)
            persistent: Whether to maintain reservation throughout framework lifetime (default: True)
        """
        self.device = device
        self.reserve_gb = reserve_gb
        self.persistent = persistent
        self.reserved_tensors = []
        self._memory_reserved = False
        self._framework_reservation_active = False

        # Register cleanup on exit
        if persistent:
            atexit.register(self._cleanup_on_exit)

    def _cleanup_on_exit(self):
        """Cleanup function registered with atexit to ensure memory is released on exit."""
        if self._memory_reserved:
            self.release_memory()

    def initialize_framework_reservation(self) -> bool:
        """
        Initialize persistent GPU memory reservation for the framework.

        This should be called when the framework starts up to establish
        baseline GPU memory reservation.

        Returns:
            True if reservation was successfully established
        """
        if not self.persistent:
            return False

        if self._framework_reservation_active:
            print("ℹ️ Framework GPU memory reservation already active")
            return True

        print("🔒 Initializing persistent GPU memory reservation for framework...")
        success = self.reserve_memory()
        if success:
            self._framework_reservation_active = True
            print("✅ Framework GPU memory reservation established")
        else:
            print("❌ Failed to establish framework GPU memory reservation")

        return success

    def prepare_for_test_execution(self) -> bool:
        """
        Temporarily release GPU memory reservation before test execution.

        This allows the test to have full access to GPU memory.

        Returns:
            True if memory was successfully released for test execution
        """
        if not self._framework_reservation_active:
            print("⚠️ No active framework reservation to release")
            return True

        print("🔓 Temporarily releasing GPU memory reservation for test execution...")
        success = self.release_memory()
        if success:
            print("✅ GPU memory released for test execution")
        else:
            print("❌ Failed to release GPU memory for test execution")

        return success

    def restore_framework_reservation(self) -> bool:
        """
        Restore GPU memory reservation immediately after test completion.

        This prevents other processes from claiming memory during framework idle time.

        Returns:
            True if framework reservation was successfully restored
        """
        if not self.persistent:
            return True

        print("🔒 Restoring GPU memory reservation after test completion...")
        success = self.reserve_memory()
        if success:
            self._framework_reservation_active = True
            print("✅ Framework GPU memory reservation restored")
        else:
            print("❌ Failed to restore framework GPU memory reservation")

        return success

    def _bytes_to_gb(self, bytes_val: int) -> float:
        """Convert bytes to gigabytes."""
        return bytes_val / (1024**3)

    def _gb_to_bytes(self, gb_val: float) -> int:
        """Convert gigabytes to bytes."""
        return int(gb_val * (1024**3))

    def get_gpu_memory_info(self) -> dict:
        """
        Get current GPU memory information.

        Returns:
            Dictionary with memory information (total, used, free in GB)
        """
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        try:
            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}

            torch.cuda.synchronize(self.device)

            total_bytes = torch.cuda.get_device_properties(self.device).total_memory
            allocated_bytes = torch.cuda.memory_allocated(self.device)
            reserved_bytes = torch.cuda.memory_reserved(self.device)

            return {
                "total_gb": self._bytes_to_gb(total_bytes),
                "allocated_gb": self._bytes_to_gb(allocated_bytes),
                "reserved_gb": self._bytes_to_gb(reserved_bytes),
                "free_gb": self._bytes_to_gb(total_bytes - allocated_bytes),
                "device": self.device,
            }
        except Exception as e:
            return {"error": f"Failed to get GPU memory info: {e}"}

    def reserve_memory(self) -> bool:
        """
        Reserve GPU memory by allocating tensors.

        Returns:
            True if memory was successfully reserved, False otherwise
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            print("⚠️ GPU memory reservation not available: PyTorch or CUDA not available")
            return False

        try:
            # Get memory info before allocation
            mem_info_before = self.get_gpu_memory_info()
            if "error" in mem_info_before:
                print(f"⚠️ Cannot reserve GPU memory: {mem_info_before['error']}")
                return False

            initial_allocated_bytes = mem_info_before.get("allocated_gb", 0) * (1024**3)
            reserve_bytes = self._gb_to_bytes(self.reserve_gb)

            print(f"🔒 Reserving {self.reserve_gb:.1f} GB GPU memory...")
            print(f"   Initial allocated memory: {mem_info_before.get('allocated_gb', 0):.2f} GB")

            # Reserve memory by allocating tensors of different sizes
            # This prevents other processes from claiming the memory
            # Use smaller chunk sizes to handle memory fragmentation better
            base_sizes = [
                128 * 1024 * 1024,  # 128MB
                64 * 1024 * 1024,  # 64MB
                32 * 1024 * 1024,  # 32MB
                16 * 1024 * 1024,  # 16MB
                8 * 1024 * 1024,  # 8MB
                4 * 1024 * 1024,  # 4MB
            ]

            allocated = 0
            remaining = reserve_bytes

            # Try to allocate in larger chunks first, then smaller ones
            for base_size in base_sizes:
                if remaining <= 0:
                    break

                # Calculate how many of this size chunk we need
                chunks_needed = remaining // base_size
                if chunks_needed == 0:
                    continue

                # Try to allocate up to chunks_needed, but be prepared to allocate fewer
                for _ in range(
                    min(chunks_needed, 10)
                ):  # Limit to 10 chunks per size to avoid too many small allocations
                    if remaining <= 0:
                        break

                    actual_size = min(base_size, remaining)

                    try:
                        # Ensure actual_size is divisible by 4 for float32
                        num_elements = actual_size // 4
                        if num_elements > 0:
                            tensor = torch.zeros(
                                num_elements, dtype=torch.float32, device=self.device
                            )
                            self.reserved_tensors.append(tensor)
                            allocated += num_elements * 4  # Actual bytes allocated for this tensor
                            remaining -= num_elements * 4
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Try smaller allocations if this size fails
                            break
                        else:
                            raise e

            # Check actual memory allocated by measuring the difference
            torch.cuda.synchronize(self.device)
            mem_info_after = self.get_gpu_memory_info()
            final_allocated_bytes = mem_info_after.get("allocated_gb", 0) * (1024**3)
            actual_reserved_bytes = final_allocated_bytes - initial_allocated_bytes
            actual_reserved_gb = actual_reserved_bytes / (1024**3)

            # Calculate tensor size for comparison
            tensor_bytes = sum(t.numel() * t.element_size() for t in self.reserved_tensors)
            tensor_gb = tensor_bytes / (1024**3)

            print(f"   Tensor data size: {tensor_gb:.2f} GB")
            print(f"   Actual GPU memory increase: {actual_reserved_gb:.2f} GB")

            # Success if we reserved at least 80% of requested amount (accounting for overhead)
            min_acceptable_gb = self.reserve_gb * 0.8
            if actual_reserved_gb >= min_acceptable_gb:
                self._memory_reserved = True
                print(
                    f"✅ Successfully reserved {actual_reserved_gb:.2f} GB GPU memory (requested: {self.reserve_gb:.2f} GB)"
                )
                return True
            else:
                print(
                    f"❌ Failed to reserve requested amount: got {actual_reserved_gb:.2f} GB, needed {min_acceptable_gb:.2f} GB (80% of {self.reserve_gb:.2f} GB)"
                )
                self.release_memory()  # Clean up partial allocations
                return False

        except Exception as e:
            print(f"⚠️ Failed to reserve GPU memory: {e}")
            self.release_memory()  # Clean up any partial allocations
            return False

    def release_memory(self) -> bool:
        """
        Release all reserved GPU memory.

        Returns:
            True if memory was released successfully
        """
        if not self._memory_reserved:
            return True

        try:
            # Delete all reserved tensors
            for tensor in self.reserved_tensors:
                del tensor

            self.reserved_tensors.clear()

            # Force garbage collection
            gc.collect()

            # Empty CUDA cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)

            self._memory_reserved = False
            print("✅ GPU memory reservation released")
            return True

        except Exception as e:
            print(f"⚠️ Error releasing GPU memory: {e}")
            return False

    def __enter__(self):
        """Context manager entry - for temporary test execution."""
        if self.persistent and self._framework_reservation_active:
            # For persistent mode, release memory for test execution
            self.prepare_for_test_execution()
        else:
            # For non-persistent mode, reserve memory
            self.reserve_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore reservation or release memory."""
        if self.persistent:
            # For persistent mode, restore framework reservation
            self.restore_framework_reservation()
        else:
            # For non-persistent mode, release memory
            self.release_memory()

    @property
    def memory_reserved(self) -> bool:
        """Check if memory is currently reserved."""
        return self._memory_reserved

    @property
    def framework_reservation_active(self) -> bool:
        """Check if framework-level persistent reservation is active."""
        return self._framework_reservation_active


def get_optimal_reserve_amount(device: int = 0) -> float:
    """
    Calculate optimal GPU memory reservation amount based on available memory.

    Args:
        device: CUDA device ID

    Returns:
        Recommended reservation amount in GB
    """
    manager = GPUMemoryManager(device=device, reserve_gb=0)
    mem_info = manager.get_gpu_memory_info()

    if "error" in mem_info:
        return 1.0  # Default fallback

    total_gb = mem_info["total_gb"]

    # Reserve 50% of total GPU memory
    optimal = total_gb * 0.5
    return optimal


# Global instance for easy access
_default_manager = None


def initialize_framework_gpu_reservation(
    device: int = 0, reserve_gb: Optional[float] = None
) -> bool:
    """
    Initialize persistent GPU memory reservation for the framework.

    This should be called at framework startup to establish baseline memory reservation.

    Args:
        device: CUDA device ID
        reserve_gb: Memory to reserve in GB (auto-calculated if None)

    Returns:
        True if framework reservation was successfully initialized
    """
    global _default_manager

    if reserve_gb is None:
        reserve_gb = get_optimal_reserve_amount(device)

    if _default_manager is None:
        _default_manager = GPUMemoryManager(device=device, reserve_gb=reserve_gb, persistent=True)

    return _default_manager.initialize_framework_reservation()


def get_gpu_memory_manager(
    device: int = 0, reserve_gb: Optional[float] = None, persistent: bool = True
) -> GPUMemoryManager:
    """
    Get or create a GPU memory manager instance.

    Args:
        device: CUDA device ID
        reserve_gb: Memory to reserve in GB (auto-calculated if None)
        persistent: Whether to use persistent reservation mode (default: True)

    Returns:
        GPUMemoryManager instance
    """
    global _default_manager

    if reserve_gb is None:
        reserve_gb = get_optimal_reserve_amount(device)

    if (
        _default_manager is None
        or _default_manager.device != device
        or _default_manager.persistent != persistent
    ):
        _default_manager = GPUMemoryManager(
            device=device, reserve_gb=reserve_gb, persistent=persistent
        )

    return _default_manager
