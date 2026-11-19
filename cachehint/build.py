"""Compile kernels at each opt level and save disassembly to build/<opt>/."""

import subprocess

from .config import BUILD_DIR, CFLAGS, KERNEL_DIR, OPT_LEVELS, WORKLOADS


def build_one(kernel: str, opt: str) -> None:
    out_dir = BUILD_DIR / opt
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = out_dir / kernel
    subprocess.run(
        ["gcc", f"-{opt}", *CFLAGS, "-o", str(binary), str(KERNEL_DIR / f"{kernel}.c")],
        check=True,
    )
    asm = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "-M", "att", str(binary)],
        check=True, capture_output=True, text=True,
    ).stdout
    (out_dir / f"{kernel}.asm").write_text(asm)


def main() -> None:
    for opt in OPT_LEVELS:
        for kernel in sorted({k for k, _ in WORKLOADS.values()}):
            build_one(kernel, opt)
            print(f"built {opt}/{kernel}")


if __name__ == "__main__":
    main()
