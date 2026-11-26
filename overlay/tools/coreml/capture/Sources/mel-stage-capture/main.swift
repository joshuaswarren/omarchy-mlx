// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Mel stage-isolation capture (CPU only — no CoreML, no model loads).
//
// Runs the pinned MelFeatureExtractor (ParakeetTDT @ 75aec2a) and a
// stepwise re-expression of its exact operations (MelFeatureExtractor.swift
// lines 98-247, MelFilterBank.swift lines 12-88), certifies that the
// stepwise final array is byte-identical to the pinned library output,
// and dumps every intermediate stage as .npy:
//
//   preemph        [480000]      float32  step 1
//   hann           [400]         float32  window constant
//   mel_fb         [128,257]     float32  filterbank rows
//   frames         [3001,512]    float32  windowed FFT inputs
//   dft_real       [3001,257]    float32  unpacked spectrum, real
//   dft_imag       [3001,257]    float32  unpacked spectrum, imag
//   power          [3001,257]    float32  sqrt-then-square
//   melproj        [3001,128]    float32  before log
//   logmel         [3001,128]    float32  after log(mel + 2^-24)
//   mean,std       [128]         float32  normalization constants
//   mel_stepwise   [3001,128]    float32  stepwise final
//   mel_pinned     [3001,128]    float32  pinned extract() final
//   manifest.json  sha256 + environment
//
// The byte-equality gate (mel_stepwise == mel_pinned) is what makes the
// intermediates a faithful record of what the pinned reference computed.
//
// Usage:
//   mel-stage-capture --waveform wave.f32 --out DIR --expect-sha256 HEX
//     (wave.f32: raw little-endian float32 PCM, at most 30 s; the pinned
//      capture's waveform.npy data section satisfies this)

import CryptoKit
import Foundation
import MelCaptureSupport
import ParakeetTDT
import Accelerate

// MARK: - npy writers (same format as parakeet-reference-capture)

func npyBytes(_ data: [Float], shape: [Int]) throws -> Data {
    var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(shape.map(String.init).joined(separator: ", "))"
    header += shape.count == 1 ? ",) }" : ") }"
    let base = 6 + 2 + 2 + header.utf8.count + 1
    let padded = ((base + 63) / 64) * 64
    header += String(repeating: " ", count: padded - base) + "\n"
    var out = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59] as [UInt8])
    out.append(contentsOf: [1, 0] as [UInt8])
    out.append(contentsOf: withUnsafeBytes(of: UInt16(header.utf8.count).littleEndian) { Array($0) })
    out.append(contentsOf: header.utf8)
    data.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
    return out
}

func dump(_ dir: URL, _ name: String, _ data: [Float], shape: [Int],
          manifest: inout [String: String]) throws {
    let bytes = try npyBytes(data, shape: shape)
    try bytes.write(to: dir.appendingPathComponent("\(name).npy"))
    manifest[name] = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
}

// MARK: - args

var wavePath: String?
var outPath: String?
var expectSha: String?
var mode = "stages"

var it = CommandLine.arguments.makeIterator()
_ = it.next()
while let arg = it.next() {
    switch arg {
    case "--waveform": wavePath = it.next()
    case "--out": outPath = it.next()
    case "--expect-sha256": expectSha = it.next()
    case "--mode": mode = it.next() ?? ""
    default:
        FileHandle.standardError.write(Data("unknown arg: \(arg)\n".utf8))
        exit(2)
    }
}
guard let outPath else {
    FileHandle.standardError.write(Data("usage: [--waveform W.f32] --out DIR [--mode stages|probe-cosf|probe-logf|probe-dft|probe-dft-core] ...\n".utf8))
    exit(2)
}

func readF32(_ path: String) -> [Float] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    precondition(data.count % 4 == 0, "input not a f32 stream")
    var out = [Float](repeating: 0, count: data.count / 4)
    _ = out.withUnsafeMutableBufferPointer { dst in
        data.withUnsafeBytes { raw in
            UnsafeMutableRawPointer(dst.baseAddress!).copyMemory(
                from: raw.baseAddress!, byteCount: data.count)
        }
    }
    return out
}

if mode == "probe-cosf" || mode == "probe-logf" {
    guard let wavePath else { exit(2) }
    let x = readF32(wavePath)
    let outURL = URL(fileURLWithPath: outPath)
    try FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)
    var manifest = [String: String]()
    if mode == "probe-cosf" {
        try dump(outURL, "cosf_out", x.map { cosf($0) }, shape: [x.count], manifest: &manifest)
    } else {
        try dump(outURL, "logf_out", x.map { logf($0) }, shape: [x.count], manifest: &manifest)
    }
    try dump(outURL, "probe_in", x, shape: [x.count], manifest: &manifest)
    let manData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys, .prettyPrinted])
    try manData.write(to: outURL.appendingPathComponent("manifest.json"))
    print("probe \(mode): \(x.count) values")
    exit(0)
}

if mode == "probe-dft" || mode == "probe-dft-core" {
    guard let wavePath else { exit(2) }
    let samples = readF32(wavePath)
    let nFFT = 512
    let nComplex = nFFT / 2
    precondition(samples.count % nFFT == 0, "dft probe needs whole 512-sample frames")
    let numFrames = samples.count / nFFT
    let coreOnly = mode == "probe-dft-core"
    let outputCount = coreOnly ? nComplex : nComplex + 1
    var realIn = [Float](repeating: 0, count: nComplex)
    var imagIn = [Float](repeating: 0, count: nComplex)
    var realOut = [Float](repeating: 0, count: nComplex)
    var imagOut = [Float](repeating: 0, count: nComplex)
    let fftSetup = coreOnly
        ? vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nComplex), .FORWARD)
        : vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD)
    guard let fftSetup else { fatalError("fft setup failed") }
    var allReal = [[Float]](), allImag = [[Float]]()
    for t in 0..<numFrames {
        let frame = Array(samples[t*nFFT..<(t+1)*nFFT])
        for k in 0..<nComplex {
            realIn[k] = frame[2 * k]
            imagIn[k] = frame[2 * k + 1]
        }
        vDSP_DFT_Execute(fftSetup, realIn, imagIn, &realOut, &imagOut)
        if coreOnly {
            allReal.append(realOut)
            allImag.append(imagOut)
        } else {
            var real = [Float](repeating: 0, count: outputCount)
            var imag = [Float](repeating: 0, count: outputCount)
            real[0] = realOut[0] * 0.5
            real[outputCount - 1] = imagOut[0] * 0.5
            for k in 1..<nComplex {
                real[k] = realOut[k] * 0.5
                imag[k] = imagOut[k] * 0.5
            }
            allReal.append(real)
            allImag.append(imag)
        }
    }
    let outURL = URL(fileURLWithPath: outPath)
    try FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)
    var manifest = [String: String]()
    let prefix = coreOnly ? "probe_core" : "probe_dft"
    func flatten(_ rows: [[Float]]) -> [Float] { rows.flatMap { $0 } }
    try dump(outURL, "\(prefix)_real", flatten(allReal), shape: [numFrames, outputCount], manifest: &manifest)
    try dump(outURL, "\(prefix)_imag", flatten(allImag), shape: [numFrames, outputCount], manifest: &manifest)
    try dump(outURL, "probe_in", samples, shape: [samples.count], manifest: &manifest)
    let manData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys, .prettyPrinted])
    try manData.write(to: outURL.appendingPathComponent("manifest.json"))
    print("probe \(mode): \(numFrames) frames")
    exit(0)
}

let inURL = URL(fileURLWithPath: wavePath!)
let waveData = try Data(contentsOf: inURL)
let inputSha = SHA256.hash(data: waveData).map { String(format: "%02x", $0) }.joined()
if let expectSha {
    guard inputSha == expectSha.lowercased() else {
        FileHandle.standardError.write(Data("input sha mismatch: want \(expectSha), got \(inputSha)\n".utf8))
        exit(3)
    }
    print("verified waveform sha256 ok")
}
guard waveData.count % 4 == 0 else { fatalError("input not a f32 stream") }
var waveform = [Float](repeating: 0, count: waveData.count / 4)
_ = waveform.withUnsafeMutableBufferPointer { dst in
    waveData.withUnsafeBytes { raw in
        UnsafeMutableRawPointer(dst.baseAddress!).copyMemory(
            from: raw.baseAddress!, byteCount: waveData.count)
    }
}
print("waveform: \(waveform.count) samples")

// MARK: - chunking (mirrors ParakeetTranscriber.transcribe / capture main.swift)

let chunkSamples = 3000 * 160
precondition(waveform.count <= chunkSamples, "stage capture covers one chunk")
var chunk = Array(waveform[0..<min(waveform.count, chunkSamples)])
if chunk.count < chunkSamples {
    chunk.append(contentsOf: [Float](repeating: 0, count: chunkSamples - chunk.count))
}

let outURL = URL(fileURLWithPath: outPath)
try FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)

// MARK: - pinned reference run (the library itself)

let pinned = try MelFeatureExtractor()
let pinnedFeatures = pinned.extract(from: chunk)
print("pinned extract: \(pinnedFeatures.numFrames) frames")

// MARK: - stepwise re-expression of MelFeatureExtractor.extract (75aec2a)
// Every operation below mirrors the pinned source call-for-call; the
// byte-equality gate at the end certifies the transcription.

let sampleRate = pinned.sampleRate, hopLength = pinned.hopLength
let winLength = pinned.winLength, nFFT = pinned.nFFT
let numMelFilters = pinned.numMelFilters
let preemphasis = pinned.preemphasis, logGuard = pinned.logGuard
let epsilon = pinned.epsilon

// --- Step 1: preemphasis (MelFeatureExtractor.swift:101-106) ---
var preemph = [Float](repeating: 0, count: chunk.count)
preemph[0] = chunk[0]
for n in 1..<chunk.count {
    preemph[n] = chunk[n] - preemphasis * chunk[n - 1]
}

// --- Step 2: center pad (108-126) ---
let padLeft = nFFT / 2, padRight = nFFT / 2
var padded = [Float](repeating: 0, count: padLeft + preemph.count + padRight)
_ = padded.withUnsafeMutableBufferPointer { buf in
    preemph.withUnsafeBufferPointer { src in
        memcpy(buf.baseAddress!.advanced(by: padLeft), src.baseAddress!,
               preemph.count * MemoryLayout<Float>.size)
    }
}
let numFrames = (padded.count - nFFT) / hopLength + 1
precondition(numFrames > 0)

// window (58-65): torch.hann_window periodic=false
var hannWindow = [Float](repeating: 0, count: winLength)
let N = Float(winLength - 1)
for n in 0..<winLength {
    hannWindow[n] = 0.5 - 0.5 * cos(2.0 * .pi * Float(n) / N)
}

// filterbank (MelFilterBank.swift:42-88, float64 math -> float32 rows)
func hzToMelSlaney(_ hz: Double) -> Double {
    let fMin = 0.0, fSp = 200.0 / 3.0, minLogHz = 1000.0
    let minLogMel = (minLogHz - fMin) / fSp
    let logstep = log(6.4) / 27.0
    return hz >= minLogHz ? minLogMel + log(hz / minLogHz) / logstep : (hz - fMin) / fSp
}
func melToHzSlaney(_ mels: Double) -> Double {
    let fSp = 200.0 / 3.0, minLogHz = 1000.0
    let minLogMel = (minLogHz - 0.0) / fSp
    let logstep = log(6.4) / 27.0
    return mels >= minLogMel ? minLogHz * exp(logstep * (mels - minLogMel)) : fSp * mels
}
let numFreqBins = nFFT / 2 + 1
let fMax = Double(sampleRate) / 2.0
let minMel = hzToMelSlaney(0.0), maxMel = hzToMelSlaney(fMax)
var melsD = [Double]()
melsD.reserveCapacity(numMelFilters + 2)
for i in 0..<(numMelFilters + 2) {
    melsD.append(minMel + (maxMel - minMel) * Double(i) / Double(numMelFilters + 1))
}
let hzBreakpoints = melsD.map { melToHzSlaney($0) }
var fftFreqs = [Double](repeating: 0, count: numFreqBins)
for k in 0..<numFreqBins { fftFreqs[k] = Double(k) * Double(sampleRate) / Double(nFFT) }
var melRows = [[Float]]()
melRows.reserveCapacity(numMelFilters)
for i in 0..<numMelFilters {
    let lower = hzBreakpoints[i], center = hzBreakpoints[i + 1], upper = hzBreakpoints[i + 2]
    let enorm = 2.0 / (upper - lower)
    var row = [Float](repeating: 0, count: numFreqBins)
    for k in 0..<numFreqBins {
        let f = fftFreqs[k]
        let lowerSlope = (f - lower) / (center - lower)
        let upperSlope = (upper - f) / (upper - center)
        let tri = max(0.0, min(lowerSlope, upperSlope))
        row[k] = Float(tri * enorm)
    }
    melRows.append(row)
}

// --- per-frame FFT path (128-216), buffers reused as pinned ---
var frameBuf = [Float](repeating: 0, count: nFFT)
var realIn = [Float](repeating: 0, count: nFFT / 2)
var imagIn = [Float](repeating: 0, count: nFFT / 2)
var realOut = [Float](repeating: 0, count: nFFT / 2)
var imagOut = [Float](repeating: 0, count: nFFT / 2)
var powerBuf = [Float](repeating: 0, count: numFreqBins)

let padOffset = (nFFT - winLength) / 2
guard let fftSetup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD) else {
    fatalError("fft setup failed")
}

var framesDump = [[Float]]()
var realDump = [[Float]](), imagDump = [[Float]](), powerDump = [[Float]]()
var melprojDump = [[Float]](), logmelDump = [[Float]]()
framesDump.reserveCapacity(numFrames)
realDump.reserveCapacity(numFrames); imagDump.reserveCapacity(numFrames)
powerDump.reserveCapacity(numFrames)
melprojDump.reserveCapacity(numFrames); logmelDump.reserveCapacity(numFrames)

for t in 0..<numFrames {
    vDSP_vclr(&frameBuf, 1, vDSP_Length(nFFT))
    let start = t * hopLength
    padded.withUnsafeBufferPointer { paddedPtr in
        let src = paddedPtr.baseAddress!.advanced(by: start)
        hannWindow.withUnsafeBufferPointer { wPtr in
            frameBuf.withUnsafeMutableBufferPointer { dstBuf in
                vDSP_vmul(src, 1, wPtr.baseAddress!, 1,
                          dstBuf.baseAddress!.advanced(by: padOffset), 1,
                          vDSP_Length(winLength))
            }
        }
    }
    framesDump.append(frameBuf)

    frameBuf.withUnsafeBufferPointer { fBuf in
        realIn.withUnsafeMutableBufferPointer { rBuf in
            imagIn.withUnsafeMutableBufferPointer { iBuf in
                for k in 0..<(nFFT / 2) {
                    rBuf[k] = fBuf[2 * k]
                    iBuf[k] = fBuf[2 * k + 1]
                }
            }
        }
    }

    vDSP_DFT_Execute(fftSetup, realIn, imagIn, &realOut, &imagOut)

    var real = [Float](repeating: 0, count: numFreqBins)
    var imag = [Float](repeating: 0, count: numFreqBins)
    real[0] = realOut[0] * 0.5
    real[numFreqBins - 1] = imagOut[0] * 0.5
    for k in 1..<(nFFT / 2) {
        real[k] = realOut[k] * 0.5
        imag[k] = imagOut[k] * 0.5
    }
    realDump.append(real); imagDump.append(imag)

    for k in 0..<numFreqBins {
        let re = real[k], im = imag[k]
        let mag = sqrtf(re * re + im * im)
        powerBuf[k] = mag * mag
    }
    powerDump.append(powerBuf)

    var melRow = [Float](repeating: 0, count: numMelFilters)
    for i in 0..<numMelFilters {
        var acc: Float = 0
        melRows[i].withUnsafeBufferPointer { mPtr in
            powerBuf.withUnsafeBufferPointer { pPtr in
                vDSP_dotpr(mPtr.baseAddress!, 1, pPtr.baseAddress!, 1, &acc, vDSP_Length(numFreqBins))
            }
        }
        melRow[i] = log(acc + logGuard)
    }
    melprojDump.append(melRow.map { $0 })  // pre-log values are not retained by the
    // pinned loop (log overwrites); melproj is reconstructed below from acc.
    logmelDump.append(melRow)
}

// The pinned loop computes log(acc + guard) immediately; to also record the
// pre-log projection we rerun the dot products once more from the identical
// power buffers (deterministic: same inputs, same Accelerate call).
for t in 0..<numFrames {
    var row = [Float](repeating: 0, count: numMelFilters)
    for i in 0..<numMelFilters {
        var acc: Float = 0
        melRows[i].withUnsafeBufferPointer { mPtr in
            powerDump[t].withUnsafeBufferPointer { pPtr in
                vDSP_dotpr(mPtr.baseAddress!, 1, pPtr.baseAddress!, 1, &acc, vDSP_Length(numFreqBins))
            }
        }
        row[i] = acc
    }
    melprojDump[t] = row
}

// --- normalization (218-243) ---
var mean = [Float](repeating: 0, count: numMelFilters)
var std = [Float](repeating: 0, count: numMelFilters)
let nT = Float(numFrames)
for t in 0..<numFrames {
    for i in 0..<numMelFilters { mean[i] += logmelDump[t][i] }
}
for i in 0..<numMelFilters { mean[i] /= nT }
let denom = max(nT - 1, 1)
for t in 0..<numFrames {
    for i in 0..<numMelFilters {
        let d = logmelDump[t][i] - mean[i]
        std[i] += d * d
    }
}
for i in 0..<numMelFilters { std[i] = Foundation.sqrt(std[i] / denom) }
var normalized = logmelDump
for t in 0..<numFrames {
    for i in 0..<numMelFilters {
        normalized[t][i] = (normalized[t][i] - mean[i]) / (std[i] + epsilon)
    }
}

// MARK: - byte-equality gate against the pinned library

func flatten(_ rows: [[Float]], cols: Int) -> [Float] {
    var flat = [Float]()
    flat.reserveCapacity(rows.count * cols)
    for r in rows { flat.append(contentsOf: r) }
    return flat
}

let stepwiseFlat = flatten(normalized, cols: numMelFilters)
let pinnedFlat = flatten(pinnedFeatures.mel, cols: numMelFilters)
precondition(stepwiseFlat.count == pinnedFlat.count)
let mismatch = MelCaptureSupport.firstFloatBitPatternMismatch(stepwiseFlat, pinnedFlat) ?? -1
guard mismatch < 0 else {
    FileHandle.standardError.write(Data("FATAL: stepwise != pinned at flat index \(mismatch)\n".utf8))
    exit(4)
}
print("gate: stepwise == pinned extract() byte-identical (\(stepwiseFlat.count) values)")

// MARK: - dump everything

var manifest = [String: String]()
try dump(outURL, "preemph", preemph, shape: [preemph.count], manifest: &manifest)
try dump(outURL, "hann", hannWindow, shape: [winLength], manifest: &manifest)
try dump(outURL, "mel_fb", flatten(melRows, cols: numFreqBins),
         shape: [numMelFilters, numFreqBins], manifest: &manifest)
try dump(outURL, "frames", flatten(framesDump, cols: nFFT),
         shape: [numFrames, nFFT], manifest: &manifest)
try dump(outURL, "dft_real", flatten(realDump, cols: numFreqBins),
         shape: [numFrames, numFreqBins], manifest: &manifest)
try dump(outURL, "dft_imag", flatten(imagDump, cols: numFreqBins),
         shape: [numFrames, numFreqBins], manifest: &manifest)
try dump(outURL, "power", flatten(powerDump, cols: numFreqBins),
         shape: [numFrames, numFreqBins], manifest: &manifest)
try dump(outURL, "melproj", flatten(melprojDump, cols: numMelFilters),
         shape: [numFrames, numMelFilters], manifest: &manifest)
try dump(outURL, "logmel", flatten(logmelDump, cols: numMelFilters),
         shape: [numFrames, numMelFilters], manifest: &manifest)
try dump(outURL, "mean", mean, shape: [numMelFilters], manifest: &manifest)
try dump(outURL, "std", std, shape: [numMelFilters], manifest: &manifest)
try dump(outURL, "mel_stepwise", stepwiseFlat, shape: [numFrames, numMelFilters], manifest: &manifest)
try dump(outURL, "mel_pinned", pinnedFlat, shape: [pinnedFeatures.numFrames, numMelFilters],
         manifest: &manifest)
try dump(outURL, "mel_mask", pinnedFeatures.attentionMask.map { Float($0) },
         shape: [pinnedFeatures.attentionMask.count], manifest: &manifest)

let env: [String: String] = [
    "reference_commit": "75aec2a1c991319657ff4dec5f602c12da6c5012",
    "input_sha256": inputSha,
    "macos": ProcessInfo.processInfo.operatingSystemVersionString,
    "swiftc_version": try MelCaptureSupport.swiftcVersion(),
    "compute": "cpu-vdsp-only",
    "byte_equality_gate": "passed",
]
let envData = try JSONSerialization.data(withJSONObject: env, options: [.sortedKeys, .prettyPrinted])
try envData.write(to: outURL.appendingPathComponent("environment.txt"))
let manData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys, .prettyPrinted])
try manData.write(to: outURL.appendingPathComponent("manifest.json"))
print("dumped \(manifest.count) stage files to \(outPath)")
