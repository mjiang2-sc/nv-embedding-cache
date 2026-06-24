#!/usr/bin/python
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Smoke tests for HostLayer with device=cpu — exercises the CPU-only inference
# path end-to-end. On a GPU-enabled host these still pass; the meaningful win is
# that they also work on a driverless system.

import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_export as nve_export
import pynve.nve as nve
import pytest
import tempfile
import torch


def _make_layer(num_embeddings, embed_size, weight, *, storage_kind):
    if storage_kind == "memblock":
        memblock = nve.UserMemBlock(weight.data_ptr())
        layer = nve_layers.NVEmbedding(
            num_embeddings, embed_size, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            storage=memblock,
            device=torch.device("cpu"),
            optimize_for_training=False,
        )
    elif storage_kind == "host_memblock":
        # Owning malloc-backed block; weight_init is copied into it by __init__.
        memblock = nve.HostMemBlock(embed_size, num_embeddings, nve.DataType_t.Float32)
        layer = nve_layers.NVEmbedding(
            num_embeddings, embed_size, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            storage=memblock,
            weight_init=weight,
            device=torch.device("cpu"),
            optimize_for_training=False,
        )
        layer._host_memblock = memblock  # keep alive
    elif storage_kind == "auto":
        layer = nve_layers.NVEmbedding(
            num_embeddings, embed_size, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            weight_init=weight,
            device=torch.device("cpu"),
            optimize_for_training=False,
        )
    else:
        raise ValueError(storage_kind)
    layer._host_weight = weight  # keep alive
    return layer


def test_host_layer_cpu_gather():
    num_embeddings = 1024
    embed_size = 8
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock")
    keys = torch.tensor([0, 5, 17, 256, 1023], dtype=torch.int64)
    out = layer(keys)
    assert out.device.type == "cpu"
    assert torch.equal(out, weight[keys])


def test_host_layer_cpu_auto_storage():
    num_embeddings = 256
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="auto")
    keys = torch.tensor([0, 13, 200, 255], dtype=torch.int64)
    out = layer(keys)
    assert out.device.type == "cpu"
    assert torch.equal(out, weight[keys])


def test_host_layer_cpu_update():
    num_embeddings = 256
    embed_size = 4
    host_weight = torch.zeros(num_embeddings, embed_size, dtype=torch.float32).contiguous()
    layer = _make_layer(num_embeddings, embed_size, host_weight, storage_kind="memblock")
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    updates = torch.arange(1.0, 1.0 + 5 * embed_size, dtype=torch.float32).reshape(5, embed_size)
    layer.update(keys, updates)
    out = layer(keys)
    assert torch.equal(out, updates)


def test_host_layer_cpu_host_memblock_gather():
    # HostMemBlock (owning, malloc-backed) used directly as HostLayer storage.
    num_embeddings = 512
    embed_size = 8
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="host_memblock")
    keys = torch.tensor([0, 5, 17, 256, 511], dtype=torch.int64)
    out = layer(keys)
    assert out.device.type == "cpu"
    assert torch.equal(out, weight[keys])


def test_host_memblock_type_tag():
    mb = nve.HostMemBlock(4, 16, nve.DataType_t.Float32)
    assert mb.get_type() == nve.MemBlockType.Host


def test_host_layer_cpu_export_load_roundtrip():
    # Exercises load_nve_layers' CPU path, which allocates a HostMemBlock internally.
    num_embeddings = 512
    embed_size = 8
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nve_layers.NVEmbedding(
                num_embeddings, embed_size, torch.float32,
                layer_type=nve_layers.LayerType.HostLayer,
                weight_init=weight, optimize_for_training=False)

    save_dir = tempfile.mkdtemp()
    nve_export.save_nve(M(), save_dir)
    layers = nve_export.load_nve_layers(save_dir, device=torch.device("cpu"))
    assert len(layers) == 1
    keys = torch.tensor([0, 5, 17, 256, 511], dtype=torch.int64)
    out = layers[0](keys)
    assert out.device.type == "cpu"
    assert torch.equal(out, weight[keys])


def test_host_layer_cpu_aot_export_load_roundtrip():
    # export_aot + load_aot on a CPU HostLayer: exercises the Python AOT path on
    # a CPU device (device_index=-1, CPU marker constant). Verifies load_aot
    # honors device='cpu' rather than forcing CUDA.
    num_embeddings = 512
    embed_size = 8
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nve_layers.NVEmbedding(
                num_embeddings, embed_size, torch.float32,
                layer_type=nve_layers.LayerType.HostLayer,
                weight_init=weight, optimize_for_training=False)

        def forward(self, keys):
            return self.emb(keys)

    keys = torch.tensor([0, 5, 17, 256, 511], dtype=torch.int64)
    with tempfile.TemporaryDirectory() as save_dir:
        nve_export.export_aot(M(), (keys,), save_dir)
        loader, layers = nve_export.load_aot(save_dir, device=torch.device("cpu"))
        out = loader.run([keys])[0]
        assert out.device.type == "cpu"
        assert torch.equal(out, weight[keys])


def test_host_layer_cpu_backprop_raises():
    # Backprop is GPU/training-only; on a host (device_id < 0) layer the binding
    # must raise a clear error rather than crashing in the CUDA runtime.
    num_embeddings = 256
    embed_size = 4
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock")
    keys = torch.tensor([1, 2, 3], dtype=torch.int64)
    grads = weight[:3].contiguous()
    with pytest.raises(Exception, match="host-only layer"):
        layer.emb_layer.concat_backprop(
            3, keys.data_ptr(), grads.data_ptr(), 0, 0, 0)


def test_host_layer_cpu_rejects_for_non_host_layer_type():
    with pytest.raises(ValueError, match="HostLayer"):
        nve_layers.NVEmbedding(
            16, 4, torch.float32,
            layer_type=nve_layers.LayerType.GPULayer,
            device=torch.device("cpu"),
            optimize_for_training=False,
        )


def test_embedding_bag_rejects_host_layer():
    # NVEmbeddingBag has no pooled-lookup path for HostLayer; must fail fast at
    # construction rather than crash at the first forward().
    with pytest.raises(ValueError, match="does not support LayerType.HostLayer"):
        nve_layers.NVEmbeddingBag(
            16, 4, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            mode="sum",
            device=torch.device("cpu"),
            optimize_for_training=False,
        )


def test_host_layer_rejects_optimize_for_training():
    # HostLayer is inference-only; constructing with the default
    # optimize_for_training=True must fail fast at __init__, not at backward time.
    with pytest.raises(ValueError, match="inference-only"):
        nve_layers.NVEmbedding(
            16, 4, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            device=torch.device("cpu"),
            # optimize_for_training defaults to True
        )


# ---------------------------------------------------------------------------
# Concurrency: per-thread execution context (Snap patch in nve_torch_ops_cpu.cpp).
#
# The CPU op originally passed /*stream=*/0, so get_exec_context(0) handed EVERY
# concurrent serving thread the SAME execution context. A context is not
# thread-safe (it holds get_buffer scratch such as linear_host_table_key_counter),
# so a multi-threaded server racing on context 0 would silently corrupt results —
# not crash. The patch keys the context by host_ctx_key() (a per-thread token), so
# each worker gets its own context. These tests drive one HostLayer binding's
# lookup from many threads at once and assert every result stays bit-exact to the
# single-thread reference. Green proves the shared-context race is gone; run
# against an UNPATCHED build they flake/mismatch under load.
# ---------------------------------------------------------------------------


def test_host_layer_cpu_concurrent_lookup_threadsafe():
    # Eager path: N threads hammer layer(keys) on ONE shared binding. The custom
    # op releases the GIL during the C++ gather, so the lookups truly overlap.
    import threading

    num_embeddings = 4096
    embed_size = 16
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock")

    num_threads = 16
    iters = 200
    # Distinct key set per thread (varied lengths/values stress the per-context
    # scratch the race corrupts).
    torch.manual_seed(0)
    key_sets = [
        torch.randint(0, num_embeddings, (1 + (t * 7) % 257,), dtype=torch.int64)
        for t in range(num_threads)
    ]
    expected = [weight[k] for k in key_sets]

    errors = []
    start = threading.Barrier(num_threads)

    def worker(t):
        keys, exp = key_sets[t], expected[t]
        try:
            start.wait()  # release all threads together → maximum contention
            for _ in range(iters):
                out = layer(keys)
                if not torch.equal(out, exp):
                    errors.append((t, "result mismatch (shared-context race?)"))
                    return
        except Exception as e:  # noqa: BLE001
            errors.append((t, repr(e)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"concurrent HostLayer lookups raced/failed: {errors[:8]}"


def test_host_layer_cpu_aot_concurrent_run_threadsafe():
    # AOT path — the production-representative case: the masterchef_v2 CPU engine
    # calls ONE AOTIModelPackageLoader's run() from multiple worker threads. Mirror
    # that: load_aot once, then hammer loader.run from N threads. The in-graph
    # nve_ops::embedding_lookup hits the same per-thread-context path.
    import threading

    num_embeddings = 4096
    embed_size = 16
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nve_layers.NVEmbedding(
                num_embeddings, embed_size, torch.float32,
                layer_type=nve_layers.LayerType.HostLayer,
                weight_init=weight, optimize_for_training=False)

        def forward(self, keys):
            return self.emb(keys)

    num_threads = 16
    iters = 100
    torch.manual_seed(1)
    key_sets = [
        torch.randint(0, num_embeddings, (1 + (t * 11) % 199,), dtype=torch.int64)
        for t in range(num_threads)
    ]
    expected = [weight[k] for k in key_sets]

    with tempfile.TemporaryDirectory() as save_dir:
        nve_export.export_aot(M(), (key_sets[0],), save_dir)
        loader, _ = nve_export.load_aot(save_dir, device=torch.device("cpu"))

        errors = []
        start = threading.Barrier(num_threads)

        def worker(t):
            keys, exp = key_sets[t], expected[t]
            try:
                start.wait()
                for _ in range(iters):
                    out = loader.run([keys])[0]
                    if not torch.equal(out, exp):
                        errors.append((t, "result mismatch (shared-context race?)"))
                        return
            except Exception as e:  # noqa: BLE001
                errors.append((t, repr(e)))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors, f"concurrent AOT HostLayer runs raced/failed: {errors[:8]}"
