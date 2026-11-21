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
        self.assertNotRegex(asm, r'OpExtInst [^\n]+\bCeil\b')
