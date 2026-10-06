# Is there an open-source path to the iOS SDK without Xcode? (research, 2026-10-07)

Verdict: **no path. Hard wall.** The Xcode download cannot be removed from the iOS instructions.
Nothing from Apple's open-source Swift work replaces the iOS SDK, and Apple's license forbids the
workaround that xtool uses on non-Apple hardware. Read the Xcode and Apple SDKs Agreement before
you tell users the setup is license-clean.

This note is research, not legal advice.

## Evidence (primary sources, fetched 2026-10-07)

1. **swift.org ships no Darwin SDK.** The Swift 6.4 install page lists three Swift SDK bundles:
   Static Linux, WebAssembly, Android. No iOS, no macOS. https://www.swift.org/install/linux/
2. **swift-sdk-generator cannot target Apple platforms.** Its table: macOS is host only; Linux and
   FreeBSD are host and target. https://github.com/swiftlang/swift-sdk-generator
3. **The iOS SDK exists only inside Xcode.** xtool's own setup asks the user for `Xcode.xip`
   ("Download Xcode 27 ... Note the path where Xcode.xip is saved") and builds the Darwin Swift SDK
   from it. xtool does not ship any Apple files. https://xtool.sh/documentation/xtool/installation-linux/
   The macOS path also needs Xcode installed: https://xtool.sh/documentation/xtool/installation-macos/
4. **UIKit, SwiftUI and the other iOS frameworks are proprietary.** No open-source replacement
   exists for modern iOS (secondary sources only, not verified in depth: Darling has no UIKit;
   touchHLE emulates only iPhoneOS 2.x-3.x apps; Chameleon is stale).

## What Apple's license says (Xcode and Apple SDKs Agreement, https://www.apple.com/legal/sla/docs/xcode.pdf)

- Header note: use of Apple Software "IS AUTHORIZED ONLY FOR EXECUTION ON AN APPLE-BRANDED PRODUCT RUNNING MACOS".
- 2.2.A: install Apple Software "on Apple-branded computers".
- 2.5: "You are expressly prohibited from separately using the Apple SDKs or attempting to run any
  part of the Apple Software on non-Apple-branded hardware."
- 2.7: "not to install, use or run the Apple Software or Apple Services on any non-Apple-branded
  computer or device"; no copying, decompiling or deriving works except as permitted.
- The Developer Program License Agreement carries the same hardware limit (2.1 and 2.6, quoted in
  the Swift forums thread https://forums.swift.org/t/xtool-cross-platform-xcode-replacement-build-ios-apps-on-linux-and-more/79803/2;
  the quote is from a forum poster and is not re-checked against Apple's page).

xtool's author says the same in that thread (post 3): xtool vends no Apple files, users must read the
license and decide if their use case complies.

## What this means for Omarchy users

| User | Hardware clause | Open point |
|------|-----------------|------------|
| x86_64 PC (Framework, etc.) | Fails: non-Apple-branded computer (2.2.A, 2.5, 2.7) | Nothing to argue: not covered |
| Apple Silicon Mac running Omarchy M+ | Meets "Apple-branded" | The header says "running macOS"; 2.5 says "separately using the Apple SDKs". A SDK extracted and used under Linux may fail both. Unresolved; only Apple or a lawyer can settle it |
| Any Mac running macOS (Xcode installed) | Meets it | xtool on macOS is the licensed route |

## Alternatives that serve the same need and are license-clean

1. Build on Apple hardware running macOS: a Mac you own (mini, laptop) or a rented Mac (MacStadium,
   AWS EC2 Mac, GitHub Actions macOS runners, Xcode Cloud). Edit on Omarchy; ship from macOS. xtool
   runs on macOS, so the same project works on both.
2. A web app or PWA: no Apple SDK at all.

## Recommendation

- Keep the `Xcode.xip` step. Do not claim "no Xcode required": the omarchy-apple-dev README title says
  "no Xcode required" while its install step needs `Xcode.xip`. Reword to "no Xcode install and no
  macOS required, but one Xcode.xip download is needed".
- Add a short license notice where the installer stops at the SDK step: it must name the Xcode and
  Apple SDKs Agreement, say that the agreement limits use to Apple-branded hardware running macOS,
  and point to option 1 above.
- Do not promise users that Apple Silicon Linux is allowed. Say it is unsettled.
- If Joshua wants certainty, ask Apple Developer Legal in writing, or ask a lawyer. That is the only
  route to a firm answer.
