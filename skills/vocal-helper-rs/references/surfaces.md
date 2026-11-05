# Surfaces: vocal-helper-rs

The Python suite's rule is that a helper exposes five surfaces: a library, two
CLIs, a FastAPI HTTP server, an MCP tool set mounted on that same app, and an
installable agent skill. This crate carries two.

## What exists

**Library**: `cargo add vocal-helper-rs`, with `features = ["onnx"]` to drive
the graphs. The whole point of the crate: speech running inside a Rust program,
an iOS or Android app, or a single binary, where Python does not go.

**Agent skill**: this directory.

## What does not exist, and why

**No CLI.** The two programs under `examples/` diarize or transcribe a WAV, but
they are examples, not an installed binary. A command line for this work exists
already: the Python twin's `vocal-helper`, with URL input, the live microphone
and summaries.

**No HTTP API.** The Python twin serves the request/response surface; this
crate is meant to run in-process, on the device.

**No MCP.** The suite's rule is that MCP is *never* a second implementation:
`fastapi-mcp` mounts `/mcp` onto the FastAPI app the helper already serves, so
every MCP tool **is** one of its routes and the two cannot drift apart. Rust
has no equivalent of that mounting, so an MCP surface here would necessarily be
a separate implementation, which is exactly the drift the rule exists to prevent.

**No paid engines, no other engines.** Gladia and pyannoteAI, whisper,
pyannote, NeMo and sherpa stay in the Python twin. This crate ports the free
local Nemotron paths only.

**No weight download.** The graphs weigh hundreds of megabytes and are not
shipped with the crate; the caller passes their path.

## Choosing between the two implementations

| Need | Reach for |
|---|---|
| Rust calling code, no interpreter | `vocal-helper-rs` |
| iOS, Android, one binary per platform | `vocal-helper-rs` |
| A command line, HTTP API, MCP tools, transcript viewer | Python `vocal-helper` |
| Live microphone, URL input, summaries, language ID | Python `vocal-helper` |
| whisper, pyannote, NeMo, sherpa, Gladia, pyannoteAI | Python `vocal-helper` |
| Downloading the graphs | Python `vocal-helper` |
