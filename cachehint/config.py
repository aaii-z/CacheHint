from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "kernels"
BUILD_DIR = ROOT / "build"
TRACE_DIR = ROOT / "data" / "traces"
RESULTS_DIR = ROOT / "results"

OPT_LEVELS = ["O0", "O2", "O3"]

# -no-pie: trace PCs match objdump addresses. -g: for addr2line only.
CFLAGS = ["-g", "-no-pie", "-fno-pie", "-fno-omit-frame-pointer"]

# workload -> (binary, args)
WORKLOADS = {
    "stream_sum": ("stream_sum", []),
    "stride_field": ("stride_field", []),
    "aos_fields": ("aos_fields", []),
    "reverse_scan": ("reverse_scan", []),
    "list_seq": ("linked_list", ["seq"]),
    "list_shuf": ("linked_list", ["shuf"]),
    "hash_probe": ("hash_probe", []),
    "matmul_naive": ("matmul_naive", []),
    "matmul_tiled": ("matmul_tiled", []),
    "spmv_csr": ("spmv_csr", []),
    "bfs": ("bfs", []),
    "stencil2d": ("stencil2d", []),
    "binsearch": ("binsearch", []),
    "histogram": ("histogram", []),
}

KERNEL_PREFIX = "k_"  # only these functions are traced
