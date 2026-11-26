// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

import Accelerate
import Foundation

struct Cursor {
    let data: Data
    var offset = 0

    mutating func readUInt32() throws -> UInt32 {
        guard offset + 4 <= data.count else { throw ProbeError.truncatedInput }
        let value = data[offset..<(offset + 4)].enumerated().reduce(UInt32(0)) {
            $0 | UInt32($1.element) << UInt32(8 * $1.offset)
        }
        offset += 4
        return value
    }

    mutating func readString() throws -> String {
        let count = Int(try readUInt32())
        guard offset + count <= data.count,
              let value = String(data: data[offset..<(offset + count)], encoding: .utf8)
        else { throw ProbeError.invalidString }
        offset += count
        return value
    }

    mutating func readFloats(count: Int) throws -> [Float] {
        try (0..<count).map { _ in Float(bitPattern: try readUInt32()) }
    }
}

enum ProbeError: Error {
    case invalidArguments
    case invalidMagic
    case invalidString
    case trailingInput
    case truncatedInput
}

struct ProbeCase {
    let name: String
    let lhs: [Float]
    let rhs: [Float]
}

func loadCases(path: String) throws -> [ProbeCase] {
    var cursor = Cursor(data: try Data(contentsOf: URL(fileURLWithPath: path)))
    guard try cursor.readUInt32() == 0x44505431 else { throw ProbeError.invalidMagic }
    let count = Int(try cursor.readUInt32())
    let cases = try (0..<count).map { _ -> ProbeCase in
        let name = try cursor.readString()
        let length = Int(try cursor.readUInt32())
        guard length > 0 else { throw ProbeError.truncatedInput }
        return ProbeCase(
            name: name,
            lhs: try cursor.readFloats(count: length),
            rhs: try cursor.readFloats(count: length)
        )
    }
    guard cursor.offset == cursor.data.count else { throw ProbeError.trailingInput }
    return cases
}

let alignments = [(0, 0), (1, 0), (0, 1), (1, 1), (4, 0), (0, 4), (4, 4)]

guard CommandLine.arguments.count == 2 else { throw ProbeError.invalidArguments }
let cases = try loadCases(path: CommandLine.arguments[1])
var report = [[String: Any]]()

for probe in cases {
    var runs = [[String: Any]]()
    for (lhsOffset, rhsOffset) in alignments {
        var lhsStorage = [Float](repeating: 0, count: probe.lhs.count + lhsOffset + 16)
        var rhsStorage = [Float](repeating: 0, count: probe.rhs.count + rhsOffset + 16)
        lhsStorage.replaceSubrange(lhsOffset..<(lhsOffset + probe.lhs.count), with: probe.lhs)
        rhsStorage.replaceSubrange(rhsOffset..<(rhsOffset + probe.rhs.count), with: probe.rhs)
        var result = Float.nan
        var lhsAddress = 0
        var rhsAddress = 0
        lhsStorage.withUnsafeBufferPointer { lhs in
            rhsStorage.withUnsafeBufferPointer { rhs in
                let lhsStart = lhs.baseAddress!.advanced(by: lhsOffset)
                let rhsStart = rhs.baseAddress!.advanced(by: rhsOffset)
                lhsAddress = Int(bitPattern: lhsStart)
                rhsAddress = Int(bitPattern: rhsStart)
                vDSP_dotpr(
                    lhsStart, 1,
                    rhsStart, 1,
                    &result,
                    vDSP_Length(probe.lhs.count)
                )
            }
        }
        runs.append([
            "lhs_offset": lhsOffset,
            "rhs_offset": rhsOffset,
            "lhs_mod64": lhsAddress & 63,
            "rhs_mod64": rhsAddress & 63,
            "result_bits": String(format: "%08x", result.bitPattern),
        ])
    }
    report.append(["name": probe.name, "length": probe.lhs.count, "runs": runs])
}

let output = try JSONSerialization.data(withJSONObject: [
    "api": "vDSP_dotpr",
    "cases": report,
], options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(output)
FileHandle.standardOutput.write(Data("\n".utf8))
