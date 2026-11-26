// swift-tools-version:5.10
// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Golden-reference capture harness for the pinned public Parakeet
// Core ML reference (phase 1). Depends on the reference implementation
// at the exact commit recorded in parakeet-reference.lock, so every
// numerics-bearing step (resampling, mel, encoder, greedy TDT decode)
// is the reference code itself.

import PackageDescription

let package = Package(
    name: "parakeet-reference-capture",
    platforms: [
        .macOS(.v14),
    ],
    dependencies: [
        .package(
            url: "https://github.com/mweinbach/parakeet-coreml-swift",
            revision: "75aec2a1c991319657ff4dec5f602c12da6c5012"
        ),
    ],
    targets: [
        .executableTarget(
            name: "parakeet-reference-capture",
            dependencies: [
                .product(name: "ParakeetTDT", package: "parakeet-coreml-swift"),
            ],
            path: "Sources/parakeet-reference-capture"
        ),
        .executableTarget(
            name: "mel-stage-capture",
            dependencies: [
                .product(name: "ParakeetTDT", package: "parakeet-coreml-swift"),
                "MelCaptureSupport",
            ],
            path: "Sources/mel-stage-capture"
        ),
        .executableTarget(
            name: "mel-dotpr-probe",
            path: "Sources/mel-dotpr-probe"
        ),
        .target(
            name: "MelCaptureSupport",
            path: "Sources/MelCaptureSupport"
        ),
        .executableTarget(
            name: "mel-capture-no-model-check",
            dependencies: ["MelCaptureSupport"],
            path: "Sources/mel-capture-no-model-check"
        ),
    ]
)
