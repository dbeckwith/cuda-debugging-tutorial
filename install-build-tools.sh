#!/usr/bin/env bash

set -euo pipefail

rustup default nightly-2018-04-27
rustup component add rust-src

if ! (cargo install --list | grep -qxP 'xargo v\d+\.\d+\.\d+:'); then
    cargo install xargo --version 0.3.12
fi

if ! (cargo install --list | grep -qxP 'ptx-linker v\d+\.\d+\.\d+:'); then
    if ! hash llvm-config-6.0 2>/dev/null; then
        echo "llvm-config-6.0 not found, installing package llvm-6.0"
        sudo apt install llvm-6.0
    fi
    LLVM_SYS_60_PREFIX=$(llvm-config-6.0 --prefix) \
    cargo install ptx-linker --version 0.6.1
fi
