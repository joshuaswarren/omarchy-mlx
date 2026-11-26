import Foundation

public enum MelCaptureSupport {
    public static func firstFloatBitPatternMismatch(_ lhs: [Float], _ rhs: [Float]) -> Int? {
        guard lhs.count == rhs.count else { return min(lhs.count, rhs.count) }
        for index in lhs.indices where lhs[index].bitPattern != rhs[index].bitPattern {
            return index
        }
        return nil
    }

    public static func swiftcVersion() throws -> String {
        let xcrun = Process()
        let swiftcPath = Pipe()
        xcrun.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        xcrun.arguments = ["--find", "swiftc"]
        xcrun.standardOutput = swiftcPath
        xcrun.standardError = FileHandle.standardError
        try xcrun.run()
        xcrun.waitUntilExit()
        guard xcrun.terminationStatus == 0 else {
            throw NSError(domain: "MelCaptureSupport", code: Int(xcrun.terminationStatus))
        }
        let executable = String(decoding: swiftcPath.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !executable.isEmpty else {
            throw NSError(domain: "MelCaptureSupport", code: 1)
        }

        let swiftc = Process()
        let version = Pipe()
        swiftc.executableURL = URL(fileURLWithPath: executable)
        swiftc.arguments = ["--version"]
        swiftc.standardOutput = version
        swiftc.standardError = FileHandle.standardError
        try swiftc.run()
        swiftc.waitUntilExit()
        guard swiftc.terminationStatus == 0 else {
            throw NSError(domain: "MelCaptureSupport", code: Int(swiftc.terminationStatus))
        }
        let output = String(decoding: version.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .split { $0.isNewline }
            .joined(separator: " | ")
        guard !output.isEmpty else {
            throw NSError(domain: "MelCaptureSupport", code: 1)
        }
        return output
    }
}
