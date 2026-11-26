import Foundation
import MelCaptureSupport

let signedZeroMismatch = MelCaptureSupport.firstFloatBitPatternMismatch([0.0], [-0.0])
precondition(signedZeroMismatch == 0, "signed-zero mismatch must be refused")
precondition(
    MelCaptureSupport.firstFloatBitPatternMismatch([1.0, -0.0], [1.0, -0.0]) == nil,
    "equal float bit patterns must pass"
)

let swiftVersion = try MelCaptureSupport.swiftcVersion()
precondition(!swiftVersion.isEmpty, "swiftc --version must produce a receipt")
print("mel-capture-no-model-check=ok swiftc=\(swiftVersion)")
