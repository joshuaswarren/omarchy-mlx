# Core ML front-end runtime modules. Installed as a python package tree
# next to the CLI programs: <prefix>/coreml/, so an installed
# mlx-omarchy-parakeet or mlx-omarchy-coreml resolves its imports the
# same way the in-repo tools do (sys.path insert of the parent
# directory). The macOS capture harness is a host-side Swift tool and
# never ships; __pycache__ is build residue.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION .
        PATTERN "__pycache__" EXCLUDE
        PATTERN "capture" EXCLUDE
        PATTERN "capture/*" EXCLUDE)
