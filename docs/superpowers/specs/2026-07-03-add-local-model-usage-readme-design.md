---
name: add-local-model-usage-section-to-readme
description: Add a "Using Local Model Weights" section to README.md consolidating --weight-path / --gguf-path / --onnx-path usage
metadata:
  type: project
  created: 2026-07-03
---

# Design: Add "Using Local Model Weights" Section to README.md

## Purpose

README.md currently has scattered local-path examples scattered across the "CLI Quick Reference" section. This design consolidates them into a single, organized section so users can easily find how to load locally-stored model weights.

## Decision

- **What**: Add a new `### Using Local Model Weights` subsection to `README.md`
- **Where**: Between "CLI Quick Reference" and "Chat"
- **How**: Pure documentation change, no code modifications

## Design Details

### Section Structure

Three subsections organized by artifact format, plus a parameter reference table:

1. **Parameters table** — one-line reference for `--weight-path`, `--artifact-format`, `--gguf-path`, `--onnx-path`, `--tokenizer-dir`, `--mmproj-path`
2. **Safetensors** — 2 examples (embedding, text)
3. **GGUF** — 3 examples (text, multimodal, OCR)
4. **ONNX** — 2 examples (embedding, OCR)
5. **Footer note** — "when using GGUF/ONNX, `--tokenizer-dir` is required"

### Content Source

All examples are reorganized from existing entries in "CLI Quick Reference" — no new examples are invented. The section does NOT duplicate examples already present in "CLI Quick Reference" but instead provides a structured reference with format-based categorization.

### Scope

- Documentation change only
- Only covers `aha run` subcommand with local paths
