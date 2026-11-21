import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SHADER = (
    Path(__file__).resolve().parents[1]
    / 'overlay/mlx/backend/omarchy/shaders/elementwise.comp'
)


@unittest.skipUnless(shutil.which('glslc') and shutil.which('spirv-dis'),
                     'requires glslc and spirv-dis')
class ElementwiseSpirvTest(unittest.TestCase):
    def disassemble(self, shader, *flags):
        with tempfile.TemporaryDirectory() as temp:
            spv = Path(temp) / 'elementwise.spv'
            subprocess.run([
                'glslc', '-O', '--target-env=vulkan1.3', *flags,
                str(shader), '-o', str(spv),
            ], check=True)
            return subprocess.check_output(['spirv-dis', str(spv)], text=True)

    def assert_rounding_width(self, asm):
        definitions = {
            result: [opcode, *args.split()]
            for result, opcode, args in re.findall(
                r'^[ \t]*(%\S+) = (Op\w+)([^\n]*)$', asm, re.MULTILINE
            )
        }

        def width(type_id):
            opcode, *args = definitions[type_id]
            if opcode == 'OpTypeVector':
                opcode, *args = definitions[args[0]]
            self.assertEqual(opcode, 'OpTypeFloat')
            return int(args[0])

        rounds = 0
        for opcode, *args in definitions.values():
            if opcode != 'OpExtInst':
                continue
            result, instruction_set, name, operand = args[:4]
            if (definitions[instruction_set] != ['OpExtInstImport', '"GLSL.std.450"']
                    or name not in {'Floor', 'Ceil', 'Trunc', 'Round', 'RoundEven'}):
                continue
            rounds += 1
            widths = (width(definitions[operand][1]), width(result))
            self.assertEqual(widths, (32, 32), f'{name}: operand/result widths {widths}')
        # SPIR-V widths cannot prove that the driver will not fold a conversion.
        return rounds

    def test_explicit_fp32_rounding_with_fp16_storage(self):
        for operand, width in [('x.value', 16), ('float(x.value)', 32)]:
            with self.subTest(width=width), tempfile.TemporaryDirectory() as temp:
                shader = Path(temp) / 'round.comp'
                shader.write_text(f'''#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) buffer Data {{ float16_t value; }} x;
void main() {{ x.value = float16_t(roundEven({operand})); }}
''')
                asm = self.disassemble(shader)
                if width == 16:
                    with self.assertRaisesRegex(AssertionError, r'RoundEven:.*\(16, 16\)'):
                        self.assert_rounding_width(asm)
                else:
                    self.assertEqual(self.assert_rounding_width(asm), 1)

    def test_fp16_storage_shader_rounds_in_fp32(self):
        for flags in [(), ('-DLOWERED_POWER=1',)]:
            with self.subTest(flags=flags):
                asm = self.disassemble(SHADER, '-DUSE_FP16=1', *flags)
                self.assertGreater(self.assert_rounding_width(asm), 0)
