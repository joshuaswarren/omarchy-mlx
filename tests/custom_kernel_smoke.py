import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import mlx.core as mx

# A whole separate process, because the disk cache exists precisely to serve a
# process that has never compiled this kernel before.
CACHE_PROBE = textwrap.dedent(
    """
    import json
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name="omarchy_spirv_cache_probe",
        input_names=["values"],
        output_names=["out"],
        source="uint i = thread_position_in_grid.x; out[i] = values[i] * 3.0f + 1.0f;",
    )
    values = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    out = kernel(
        inputs=[values],
        output_shapes=[(4,)],
        output_dtypes=[mx.float32],
        grid=(4, 1, 1),
        threadgroup=(4, 1, 1),
        stream=mx.gpu,
    )[0]
    mx.eval(out)
    print(json.dumps(out.tolist()))
    """
)


class CustomKernelSmoke(unittest.TestCase):
    def call(self, kernel, inputs, shape, dtype, *, grid=None, threadgroup=None, **kwargs):
        size = 1
        for dimension in shape:
            size *= dimension
        return kernel(
            inputs=inputs,
            output_shapes=[shape],
            output_dtypes=[dtype],
            grid=grid or (size, 1, 1),
            threadgroup=threadgroup or (min(size, 32), 1, 1),
            stream=mx.gpu,
            **kwargs,
        )[0]


    def test_translation_cache_keyed_by_translator_source(self):
        """A .tr entry carries the MLXOTR1 magic and the current version, and
        its filename is the hash of an identity that includes the translator
        source hash: a rebuilt binary with changed translation logic gets a
        different name and never reads the previous binary's entry."""
        import hashlib
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, MLX_OMARCHY_SPIRV_CACHE=cache)
            probe = textwrap.dedent(
                """
                import mlx.core as mx
                kernel = mx.fast.metal_kernel(
                    name="omarchy_translation_cache_probe",
                    input_names=["values"],
                    output_names=["out"],
                    source="uint i = thread_position_in_grid.x; out[i] = values[i] + 1.0f;",
                )
                values = mx.array([1.0, 2.0], dtype=mx.float32)
                out = kernel(inputs=[values], output_shapes=[(2,)],
                             output_dtypes=[mx.float32], grid=(2, 1, 1),
                             threadgroup=(2, 1, 1), stream=mx.gpu)[0]
                mx.eval(out)
                print(out.tolist())
                """
            )
            first = subprocess.run(
                [sys.executable, "-c", probe], env=env,
                capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("[2.0, 3.0]", first.stdout)
            entries = sorted(Path(cache).glob("*.tr"))
            self.assertTrue(entries, "no .tr translation entry was written")
            blob = entries[0].read_bytes()
            self.assertTrue(blob.startswith(b"MLXOTR1"))
            # The filename is sha256(identity + cache-version + translator
            # source hash): 64 hex chars. Rebuilding with changed translator
            # source changes that hash, so the previous binary's entry is
            # never read.
            self.assertRegex(entries[0].name, r"^[0-9a-f]{64}\.tr$")

    def test_msl_body_runs_on_gpu(self):
        kernel = mx.fast.metal_kernel(
            name="omarchy_affine",
            input_names=["values", "scale"],
            output_names=["out"],
            source="""
                uint index = thread_position_in_grid.x;
                out[index] = values[index] * scale + 3.0f;
            """,
        )
        values = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        out = self.call(kernel, [values, 2.0], values.shape, values.dtype)
        self.assertEqual(out.tolist(), [5.0, 7.0, 9.0, 11.0])

    def test_templates_scalars_bfloat_and_multiple_outputs(self):
        kernel = mx.fast.metal_kernel(
            name="omarchy_arguments",
            input_names=["a", "b", "c", "d"],
            output_names=["out1", "out2"],
            source="""
                uint elem = thread_position_in_grid.x;
                T tmp = a[0];
                if (enabled) {
                    out1[elem] = a[1] + b[2] + c[3] + d + extra;
                } else {
                    out1[elem] = tmp;
                }
                out2[elem] = a[1] + b[2] + c[1] - d;
            """,
        )
        out1, out2 = kernel(
            inputs=[
                mx.array([1.0, 2.0]),
                mx.array([3, 4, 5]),
                mx.array([6.0, 7.0, 8.0, 9.0], dtype=mx.bfloat16),
                2.0,
            ],
            template=[("enabled", True), ("extra", 3), ("T", mx.float16)],
            grid=(4, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=[(4,), (4,)],
            output_dtypes=[mx.float32, mx.int32],
            stream=mx.gpu,
        )
        self.assertEqual(out1.tolist(), [21.0] * 4)
        self.assertEqual(out2.tolist(), [12] * 4)

    def test_noncontiguous_shape_and_stride_metadata(self):
        values = mx.arange(12, dtype=mx.float32).reshape(3, 4).T
        kernel = mx.fast.metal_kernel(
            name="omarchy_strides",
            input_names=["inp"],
            output_names=["out"],
            ensure_row_contiguous=False,
            source="""
                uint elem = thread_position_in_grid.x;
                uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
                out[elem] = inp[loc];
            """,
        )
        out = self.call(kernel, [values], values.shape, values.dtype)
        self.assertEqual(out.tolist(), values.tolist())

    def test_header_helper_and_threadgroup_attribute(self):
        helper = mx.fast.metal_kernel(
            name="omarchy_helper",
            input_names=["values"],
            output_names=["out"],
            header="""
                template <typename T>
                T twice(T value) { return value + value; }
            """,
            source="""
                uint elem = thread_position_in_grid.x;
                out[elem] = twice(values[elem]);
            """,
        )
        values = mx.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(
            self.call(helper, [values], values.shape, values.dtype).tolist(),
            [2.0, 4.0, 6.0, 8.0],
        )

        attribute = mx.fast.metal_kernel(
            name="omarchy_attribute",
            input_names=["values"],
            output_names=["result"],
            source="result[0] = threads_per_threadgroup.x;",
        )
        result = self.call(
            attribute,
            [values],
            (1,),
            mx.uint32,
            grid=(2, 1, 1),
            threadgroup=(2, 1, 1),
        )
        self.assertEqual(result.item(), 2)

    def test_same_name_different_source_in_one_batch(self):
        values = mx.arange(16, dtype=mx.float32)

        def apply(source):
            kernel = mx.fast.metal_kernel(
                name="omarchy_cache_key",
                input_names=["values"],
                output_names=["out"],
                source=source,
            )
            return self.call(kernel, [values], values.shape, values.dtype)

        doubled = apply(
            "uint elem = thread_position_in_grid.x; out[elem] = values[elem] * 2.0f;"
        )
        shifted = apply(
            "uint elem = thread_position_in_grid.x; out[elem] = values[elem] + 100.0f;"
        )
        mx.eval(doubled, shifted)
        self.assertEqual(doubled.tolist(), (values * 2).tolist())
        self.assertEqual(shifted.tolist(), (values + 100).tolist())

    def test_math_mode_and_mixed_dtypes(self):
        mode_source = """
            uint elem = thread_position_in_grid.x;
            #if defined(__FAST_MATH__) && __FAST_MATH__
            out[elem] = 1.0f;
            #else
            out[elem] = 0.0f;
            #endif
        """
        values = mx.zeros((4,), dtype=mx.float32)
        for mode, expected in (("safe", [0.0] * 4), ("fast", [1.0] * 4)):
            kernel = mx.fast.metal_kernel(
                name="omarchy_math_mode",
                input_names=["values"],
                output_names=["out"],
                source=mode_source,
                compile_options={"math_mode": mode},
            )
            self.assertEqual(
                self.call(kernel, [values], values.shape, values.dtype).tolist(),
                expected,
            )

        mixed = mx.fast.metal_kernel(
            name="omarchy_mixed_dtypes",
            input_names=["values"],
            output_names=["out"],
            source="""
                uint elem = thread_position_in_grid.x;
                out[elem] = values[elem] + values[elem];
            """,
        )
        half = mx.full((8,), 1.5, dtype=mx.float16)
        single = mx.full((8,), 2.5, dtype=mx.float32)
        total = self.call(mixed, [half], half.shape, half.dtype).astype(mx.float32)
        total = total + self.call(mixed, [single], single.shape, single.dtype)
        self.assertEqual(total.tolist(), [8.0] * 8)

    def test_threadgroup_memory_and_atomic_output(self):
        shared = mx.fast.metal_kernel(
            name="omarchy_shared",
            input_names=["values"],
            output_names=["out"],
            source="""
                threadgroup float scratch[4];
                uint lane = thread_position_in_threadgroup.x;
                scratch[lane] = values[lane];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) {
                    out[0] = scratch[0] + scratch[1] + scratch[2] + scratch[3];
                }
            """,
        )
        values = mx.array([1.0, 2.0, 3.0, 4.0])
        result = self.call(
            shared,
            [values],
            (1,),
            mx.float32,
            grid=(4, 1, 1),
            threadgroup=(4, 1, 1),
        )
        self.assertEqual(result.item(), 10.0)

        atomic = mx.fast.metal_kernel(
            name="omarchy_atomic",
            input_names=["values"],
            output_names=["out"],
            atomic_outputs=True,
            source="""
                uint elem = thread_position_in_grid.x;
                atomic_fetch_add_explicit(
                    &out[0], values[elem], memory_order_relaxed);
            """,
        )
        result = self.call(
            atomic,
            [values],
            (1,),
            mx.float32,
            grid=(4, 1, 1),
            threadgroup=(4, 1, 1),
            init_value=0.0,
        )
        self.assertEqual(result.item(), 10.0)

    def test_unsupported_msl_is_named_refusal(self):
        kernel = mx.fast.metal_kernel(
            name="unsupported_texture",
            input_names=["values"],
            output_names=["out"],
            source="texture2d<float> image; out[0] = values[0];",
        )
        out = self.call(
            kernel,
            [mx.array([1.0])],
            (1,),
            mx.float32,
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )
        with self.assertRaisesRegex(RuntimeError, "fast::CustomKernel MSL subset"):
            mx.eval(out)

    def probe(self, cache, cwd):
        environment = dict(os.environ, MLX_OMARCHY_SPIRV_CACHE=str(cache))
        finished = subprocess.run(
            [sys.executable, "-c", CACHE_PROBE],
            env=environment,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(finished.returncode, 0, finished.stderr)
        return json.loads(finished.stdout.strip().splitlines()[-1])

    def test_spirv_cache_hit_serves_the_cold_compile_byte_for_byte(self):
        expected = [4.0, 7.0, 10.0, 13.0]
        with tempfile.TemporaryDirectory() as root:
            first_cache = Path(root) / "first"
            second_cache = Path(root) / "second"

            first = self.probe(first_cache, root)
            entries = sorted(first_cache.glob("*.spv"))
            self.assertEqual(len(entries), 1)
            compiled = entries[0].read_bytes()

            # An independent cold compile lands on the same entry name with the
            # same bytes, so the key names the compilation rather than the run.
            second = self.probe(second_cache, root)
            repeated = sorted(second_cache.glob("*.spv"))
            self.assertEqual([p.name for p in repeated], [entries[0].name])
            self.assertEqual(repeated[0].read_bytes(), compiled)

            stamp = entries[0].stat().st_mtime_ns
            third = self.probe(first_cache, root)
            self.assertEqual(entries[0].stat().st_mtime_ns, stamp)

            self.assertEqual([first, second, third], [expected] * 3)

    def test_spirv_cache_is_disabled_by_the_environment(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(self.probe("0", root), [4.0, 7.0, 10.0, 13.0])
            self.assertEqual(sorted(Path(root).iterdir()), [])
    def test_one_source_dispatched_at_several_shapes(self):
        """Translation is cached, and the launch geometry is part of what it is.

        The generated GLSL carries the grid bounds in its entry guard and the
        group size in its layout, so a cache keyed on the source alone would
        serve a kernel compiled for one shape to a dispatch of another, leaving
        the tail of the larger output unwritten.
        """
        kernel = mx.fast.metal_kernel(
            name="omarchy_shape_reuse",
            input_names=["values"],
            output_names=["out"],
            source="uint i = thread_position_in_grid.x; out[i] = values[0] + float(i);",
        )
        for size, group in ((8, 4), (64, 32), (8, 8), (64, 8)):
            out = self.call(
                kernel,
                [mx.array([1.0])],
                (size,),
                mx.float32,
                grid=(size, 1, 1),
                threadgroup=(group, 1, 1),
            )
            mx.eval(out)
            self.assertEqual(
                out.tolist(),
                [1.0 + index for index in range(size)],
                f"size {size}, group {group}",
            )


if __name__ == "__main__":
    unittest.main()
