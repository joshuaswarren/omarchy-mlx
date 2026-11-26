// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Golden-reference capture for the pinned public Parakeet Core ML
// reference. Every numerics-bearing step (AVAudioConverter resample,
// vDSP mel, Core ML encoder, greedy TDT decode) runs inside the pinned
// reference library (ParakeetTDT at the exact SwiftPM revision); this
// executable only orchestrates and dumps tensors.
//
// Usage:
//   parakeet-reference-capture \
//     --audio /path/jfk.flac \
//     --models /path/model-dir          (encoder/decoder/joint.mlpackage + tokenizer.json)
//     --out   /path/capture-dir
//     --compute-units ane|gpu|cpu|all   (default ane)
//     --expect PATH=SHA256 ...          (verified before any inference)

import CoreML
import CryptoKit
import Foundation
import ParakeetTDT

// MARK: - npy writer

func writeNpy(_ path: URL, data: [Float]) throws {
    try writeNpy(path, bytes: data.withUnsafeBufferPointer { Data(buffer: $0) },
                 descr: "<f4", shape: [data.count])
}

func writeNpy(_ path: URL, int32 data: [Int32]) throws {
    try writeNpy(path, bytes: data.withUnsafeBufferPointer { Data(buffer: $0) },
                 descr: "<i4", shape: [data.count])
}

func writeNpy2d(_ path: URL, rows: [[Float]], cols: Int) throws {
    var flat = [Float]()
    flat.reserveCapacity(rows.count * cols)
    for r in rows { flat.append(contentsOf: r) }
    try writeNpy(path, bytes: flat.withUnsafeBufferPointer { Data(buffer: $0) },
                 descr: "<f4", shape: [rows.count, cols])
}

func writeNpy(_ path: URL, mlArray: MLMultiArray) throws {
    var shape = mlArray.shape.map { $0.intValue }
    if shape.isEmpty { shape = [1] }
    let descr: String
    switch mlArray.dataType {
    case .float32: descr = "<f4"
    case .int32:   descr = "<i4"
    case .float16: descr = "<f2"
    default: throw CaptureError.unsupportedDtype(mlArray.dataType)
    }
    let itemSize = mlArray.dataType == .float16 ? 2 : 4
    let strides = mlArray.strides.map { $0.intValue }
    let source = mlArray.dataPointer.assumingMemoryBound(to: UInt8.self)
    var bytes = Data(count: mlArray.count * itemSize)
    bytes.withUnsafeMutableBytes { destination in
        for linear in 0..<mlArray.count {
            var remaining = linear
            var offset = 0
            for axis in shape.indices.reversed() {
                offset += (remaining % shape[axis]) * strides[axis]
                remaining /= shape[axis]
            }
            destination.baseAddress!.advanced(by: linear * itemSize)
                .copyMemory(from: source.advanced(by: offset * itemSize), byteCount: itemSize)
        }
    }
    try writeNpy(path, bytes: bytes, descr: descr, shape: shape)
}

func writeNpy(_ path: URL, bytes: Data, descr: String, shape: [Int]) throws {
    var header = "{'descr': '\(descr)', 'fortran_order': False, 'shape': (\(shape.map(String.init).joined(separator: ", "))"
    header += shape.count == 1 ? ",) }" : ") }"
    // pad header so magic(6) + version(2) + len(2) + header is a multiple of 64
    let base = 6 + 2 + 2 + header.utf8.count + 1
    let padded = ((base + 63) / 64) * 64
    header += String(repeating: " ", count: padded - base) + "\n"
    var out = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59] as [UInt8])  // \x93NUMPY
    out.append(contentsOf: [1, 0] as [UInt8])                        // version 1.0
    out.append(contentsOf: withUnsafeBytes(of: UInt16(header.utf8.count).littleEndian) { Array($0) })
    out.append(contentsOf: header.utf8)
    out.append(bytes)
    try out.write(to: path)
}

enum CaptureError: Error, CustomStringConvertible {
    case unsupportedDtype(MLMultiArrayDataType)
    case missingExpect(String)
    case badExpectFormat(String)

    var description: String {
        switch self {
        case .unsupportedDtype(let d): return "unsupported npy dtype: \(d.rawValue)"
        case .missingExpect(let p): return "expected file missing: \(p)"
        case .badExpectFormat(let s): return "--expect expects PATH=SHA256, got: \(s)"
        }
    }
}

// MARK: - args

var audioPath: String?
var modelsPath: String?
var outPath: String?
var computeUnits = "ane"
var expects: [(String, String)] = []

var it = CommandLine.arguments.makeIterator()
_ = it.next()
while let arg = it.next() {
    switch arg {
    case "--audio": audioPath = it.next()
    case "--models": modelsPath = it.next()
    case "--out": outPath = it.next()
    case "--compute-units": computeUnits = it.next() ?? ""
    case "--expect":
        guard let spec = it.next(), let eq = spec.firstIndex(of: "=") else {
            throw CaptureError.badExpectFormat("missing '='")
        }
        expects.append((String(spec[..<eq]), String(spec[spec.index(after: eq)...])))
    default:
        throw CaptureError.badExpectFormat(arg)
    }
}

guard let audioPath, let modelsPath, let outPath else {
    FileHandle.standardError.write(Data("usage: --audio A --models M --out O [--compute-units ane] [--expect P=S]...\n".utf8))
    exit(2)
}

guard let selectedComputeUnits = ParakeetComputeUnits(rawValue: computeUnits) else {
    FileHandle.standardError.write(Data("--compute-units must be ane, gpu, cpu, or all\n".utf8))
    exit(2)
}

let modelsURL = URL(fileURLWithPath: modelsPath)
let outURL = URL(fileURLWithPath: outPath)
try FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)

// MARK: - verify pins before any inference

for (rel, want) in expects {
    let f = modelsURL.appendingPathComponent(rel)
    guard FileManager.default.fileExists(atPath: f.path) else {
        throw CaptureError.missingExpect(f.path)
    }
    let data = try Data(contentsOf: f)
    let got = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    guard got == want.lowercased() else {
        FileHandle.standardError.write(Data("sha mismatch for \(rel): want \(want), got \(got)\n".utf8))
        exit(3)
    }
    print("verified \(rel) sha256 ok")
}

// MARK: - environment receipt

func sysctl(_ name: String) -> String {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    var buf = [CChar](repeating: 0, count: size)
    sysctlbyname(name, &buf, &size, nil, 0)
    return String(cString: buf)
}

let coreMLVersion = Bundle(url: URL(fileURLWithPath: "/System/Library/Frameworks/CoreML.framework"))?
    .infoDictionary?["CFBundleVersion"] as? String ?? "unknown"

let environment: [String: String] = [
    "chip": sysctl("machdep.cpu.brand_string"),
    "hw_model": sysctl("hw.model"),
    "macos_release": "macOS \(ProcessInfo.processInfo.operatingSystemVersionString)",
    "os_build": sysctl("kern.osproductversion") + " (" + sysctl("kern.osversion") + ")",
    "coreml_framework_version": coreMLVersion,
    "swift_toolchain": "recorded-by-wrapper",
    "compute_units": computeUnits,
    "reference_commit": "75aec2a1c991319657ff4dec5f602c12da6c5012",
    "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
]

// MARK: - audio load (reference AudioLoader: AVAudioConverter -> 16k mono)

let audioURL = URL(fileURLWithPath: audioPath)
let waveform = try AudioLoader.loadMono16k(at: audioURL)
try writeNpy(outURL.appendingPathComponent("waveform.npy"), data: waveform)
print("waveform: \(waveform.count) samples @16kHz (\(String(format: "%.2f", Double(waveform.count) / 16000.0))s)")

// MARK: - chunking (mirrors ParakeetTranscriber.transcribe(samples:))

let chunkSamples = 3000 * 160
var chunks: [[Float]] = []
var cursor = 0
while cursor < waveform.count {
    let end = min(cursor + chunkSamples, waveform.count)
    var chunk = Array(waveform[cursor..<end])
    if chunk.count < chunkSamples {
        chunk.append(contentsOf: [Float](repeating: 0, count: chunkSamples - chunk.count))
    }
    chunks.append(chunk)
    cursor += chunkSamples
}
print("chunks: \(chunks.count)")
guard chunks.count == 1 else {
    FileHandle.standardError.write(Data("reference capture requires one nonempty audio chunk (at most 30 seconds)\n".utf8))
    exit(2)
}

// MARK: - mel (reference MelFeatureExtractor)

let extractor = try MelFeatureExtractor()
let features = extractor.extract(from: chunks[0])
try writeNpy2d(outURL.appendingPathComponent("mel.npy"), rows: features.mel, cols: 128)
try writeNpy(outURL.appendingPathComponent("mel_mask.npy"), int32: features.attentionMask)
print("mel: \(features.numFrames) frames x 128 bins")

// MARK: - Core ML models + runner (same construction as ParakeetTranscriber)

let config = MLModelConfiguration()
config.computeUnits = selectedComputeUnits.mlComputeUnits

func loadModel(_ name: String) throws -> MLModel {
    // Same compile step the reference ModelCache performs (no-numerics
    // transform); macOS 26 MLModel(contentsOf:) requires a compiled bundle.
    let src = modelsURL.appendingPathComponent("\(name).mlpackage")
    let compiled = try MLModel.compileModel(at: src)
    return try MLModel(contentsOf: compiled, configuration: config)
}

let encoder = try loadModel("encoder")
let decoder = try loadModel("decoder")
let joint = try loadModel("joint")
let tokenizer = try Tokenizer(tokenizerJSONURL: modelsURL.appendingPathComponent("tokenizer.json"))

// Encoder input construction mirrors ModelRunner.runEncoder: zero-filled
// [1,3000,128] f32 features + [1,3000] i32 mask, valid frames copied in.
let featuresArray = try MLMultiArray(shape: [1, 3000, 128], dataType: .float32)
let maskArray = try MLMultiArray(shape: [1, 3000], dataType: .int32)
do {
    let fPtr = featuresArray.dataPointer.assumingMemoryBound(to: Float.self)
    memset(fPtr, 0, 3000 * 128 * MemoryLayout<Float>.size)
    for ti in 0..<min(features.numFrames, 3000) {
        let row = features.mel[ti]
        row.withUnsafeBufferPointer { src in
            memcpy(fPtr.advanced(by: ti * 128), src.baseAddress!,
                   min(row.count, 128) * MemoryLayout<Float>.size)
        }
    }
    let mPtr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
    memset(mPtr, 0, 3000 * MemoryLayout<Int32>.size)
    for ti in 0..<min(features.attentionMask.count, 3000) {
        mPtr[ti] = features.attentionMask[ti]
    }
}

let encoderInputs = try MLDictionaryFeatureProvider(dictionary: [
    "input_features": MLFeatureValue(multiArray: featuresArray),
    "attention_mask": MLFeatureValue(multiArray: maskArray),
])
let encoderOut = try encoder.prediction(from: encoderInputs)
guard let hidden = encoderOut.featureValue(for: "encoder_hidden")?.multiArrayValue
else { fatalError("missing encoder_hidden output") }
guard let mask = encoderOut.featureValue(for: "encoder_mask")?.multiArrayValue
else { fatalError("missing encoder_mask output") }

// the actual encoder inputs as fed (zero-padded to [1,3000,128] / [1,3000])
try writeNpy(outURL.appendingPathComponent("encoder_input_features.npy"), mlArray: featuresArray)
try writeNpy(outURL.appendingPathComponent("encoder_input_mask.npy"), mlArray: maskArray)
try writeNpy(outURL.appendingPathComponent("encoder_hidden.npy"), mlArray: hidden)
try writeNpy(outURL.appendingPathComponent("encoder_mask.npy"), mlArray: mask)
print("encoder_hidden: \(hidden.shape.map { $0.intValue }) dtype=\(hidden.dataType == .float32 ? "f32" : "other")")

// MARK: - greedy TDT decode (reference loop)

let decWorker = try DecoderWorker(
    decoder: decoder, joint: joint, decoderHiddenLayers: 2, decoderHiddenSize: 640
)
let decoded = try GreedyTDTDecoder.decode(
    encoderHidden: hidden, encoderMask: mask, worker: decWorker,
    blankTokenId: 8192, durations: [0, 1, 2, 3, 4], maxSymbolsPerStep: 10
)
let transcript = tokenizer.decode(decoded.tokenIds, skipSpecial: true)

let tokenPayload: [String: Any] = [
    "token_ids": decoded.tokenIds,
    "frame_indices": decoded.frameIndices,
    "durations": decoded.durations,
]
let tokenData = try JSONSerialization.data(withJSONObject: tokenPayload, options: [.prettyPrinted, .sortedKeys])
try tokenData.write(to: outURL.appendingPathComponent("token_ids.json"))
try transcript.write(to: outURL.appendingPathComponent("transcript.txt"), atomically: true, encoding: .utf8)
print("tokens: \(decoded.tokenIds.count)  transcript: \(transcript)")

// MARK: - cross-check with the reference end-to-end transcriber

let transcriber = try ParakeetTranscriber(
    modelsRoot: modelsURL, computeUnits: selectedComputeUnits,
    decoderWorkers: 1
)
let e2e = try transcriber.transcribe(audioURL: audioURL)
let crosscheck: [String: Any] = [
    "tokens_match": e2e.tokenIds == decoded.tokenIds,
    "transcript_match": e2e.text == transcript,
    "e2e_transcript": e2e.text,
    "e2e_token_count": e2e.tokenIds.count,
]
let ccData = try JSONSerialization.data(withJSONObject: crosscheck, options: [.prettyPrinted, .sortedKeys])
try ccData.write(to: outURL.appendingPathComponent("crosscheck.json"))
print("crosscheck: tokens_match=\(e2e.tokenIds == decoded.tokenIds) transcript_match=\(e2e.text == transcript)")
guard e2e.tokenIds == decoded.tokenIds && e2e.text == transcript else {
    FileHandle.standardError.write(Data("reference capture crosscheck failed; outputs are not a golden reference\n".utf8))
    exit(4)
}

// MARK: - environment + manifest

let envData = try JSONSerialization.data(withJSONObject: environment, options: [.prettyPrinted, .sortedKeys])
try envData.write(to: outURL.appendingPathComponent("environment.json"))

let manifestFiles = try FileManager.default.contentsOfDirectory(at: outURL, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension != "sha256" && $0.lastPathComponent != "manifest.sha256" }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }
var manifest = ""
for f in manifestFiles {
    let data = try Data(contentsOf: f)
    let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    manifest += "\(sha)  \(f.lastPathComponent)\n"
}
try manifest.write(to: outURL.appendingPathComponent("manifest.sha256"), atomically: true, encoding: .utf8)
print("manifest written for \(manifestFiles.count) files")
