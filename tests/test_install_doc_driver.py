"""Install docs teach the private Honeykrisp ICD.

The supported driver is omarchy-mlx-vulkan under /usr/lib/omarchy-mlx.
The install article must not send readers through a system Mesa replacement.
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "docs" / "install-omarchy.md"
README = ROOT / "README.md"

PACKAGED_ICD = "/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json"
PACKAGED_SHA = "/usr/lib/omarchy-mlx/vulkan/mesa-git-sha"


class InstallDocDriverTests(unittest.TestCase):
    def test_install_doc_teaches_the_packaged_icd(self):
        text = INSTALL.read_text()
        self.assertIn("omarchy-mlx-vulkan", text)
        self.assertIn(PACKAGED_ICD, text)
        self.assertIn(PACKAGED_SHA, text)
        self.assertIn("icd_source", text)
        self.assertIn("OMARCHY_MLX_SYSTEM_PREFIX", text)
        self.assertIn("/etc/vulkan/icd.d", text)
        self.assertIn("MLX_OMARCHY_EXPECTED_HK_SHA", text)
        lowered = text.lower()
        self.assertNotIn("asahi", lowered)
        self.assertNotIn("remove mesa", lowered)
        self.assertNotIn("pacman -u mesa", lowered)

    def test_readme_points_at_the_packaged_icd(self):
        text = README.read_text()
        self.assertIn("omarchy-mlx-vulkan", text)
        self.assertIn(PACKAGED_ICD, text)
        self.assertIn("does not install `omarchy-mlx-vulkan`", text)
        self.assertNotIn("fork driver", text.lower())
        self.assertNotIn("asahi", text.lower())

    def test_pkgbuild_example_depends_on_the_vulkan_package(self):
        text = (ROOT / "packaging" / "PKGBUILD.example").read_text()
        self.assertIn("omarchy-mlx-vulkan=$pkgver-$pkgrel", text)
        self.assertIn(PACKAGED_ICD, text)
        self.assertNotIn("asahi", text.lower())


if __name__ == "__main__":
    unittest.main()
