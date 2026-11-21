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
    def test_fp16_pipeline_does_not_require_native_ceil(self):
        with tempfile.TemporaryDirectory() as temp:
            spv = Path(temp) / 'elementwise.spv'
            subprocess.run([
                'glslc', '-O', '--target-env=vulkan1.3', '-DUSE_FP16=1',
                str(SHADER), '-o', str(spv),
            ], check=True)
            asm = subprocess.check_output(['spirv-dis', str(spv)], text=True)
        # G18G AGX3 packer refusals (agx3_pack_binary.c 14822/19737): no
        # grounded encoding for these with a 16-bit operand. The module must
        # lower them without the reserved set; the operand-width fusion is
        # guarded by the hardware parity run, not this static check.
        self.assertNotRegex(asm, r'OpExtInst [^\n]+\b(Ceil|Trunc|Round|RoundEven)\b')
