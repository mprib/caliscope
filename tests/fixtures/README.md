# Test Fixtures

Small binary artifacts used by tests, with generator scripts for provenance.

## onnx/

Tiny constant-output ONNX models for testing the OnnxTracker pipeline.
Each model has a generator script (requires `onnx` dev dependency) and a
TOML model card. The `.onnx` binaries are checked into git so tests run
without the `onnx` package.
