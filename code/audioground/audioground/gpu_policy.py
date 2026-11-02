"""Sets CUDA_DEVICE_ORDER=PCI_BUS_ID before torch initialises CUDA, so that CUDA indices match nvidia-smi."""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def enforce(device="cuda"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
