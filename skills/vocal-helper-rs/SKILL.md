---
name: vocal-helper-rs
description: >-
  Speech diarization and transcription in Rust, with no Python and no torch: the free local Nemotron paths of vocal-helper. NemotronDiar streams 16 kHz audio through the Nemotron 3 Diarization ONNX graph and returns who-spoke-when turns or per-speaker probabilities every 10 ms; NemotronAsr runs the Nemotron 3.5 graph and returns timed words, speaker and event tags. Both sit behind the onnx feature (ort); the pure parts (log-mel, speaker cache, streaming schedule, CTC decoding, ONNX metadata) build without it. Use when the user is writing Rust and says "diarize this in Rust", "who spoke when without Python", "transcribe on device", "speech on iOS / Android", "Nemotron from Rust", or names cargo add vocal-helper-rs or vocal_helper_rs. Do NOT use for a CLI, HTTP API, MCP, live microphone, URL input, summaries or paid engines (Gladia, pyannoteAI): the Python twin vocal-helper covers those. Do NOT use to capture a mic (capture-helper-rs), to decode or resample files (audio-helper-rs), or to synthesize speech.
license: BSD-3-Clause
metadata:
  author: Warith Harchaoui
  contact: warithmetics@deraison.ai
  repository: https://github.com/warith-harchaoui/vocal-helper-rs
  crate: https://crates.io/crates/vocal-helper-rs
  language: rust
---

# vocal-helper-rs

The free local Nemotron paths of the Python `vocal-helper`, from Rust: streaming
diarization (Nemotron 3) and transcription (Nemotron 3.5), each through its
ONNX graph. A library only; the graphs are not shipped, the caller passes
their path.

## The surface

| Intent | Library | Feature |
|--------|---------|---------|
| Who spoke when | `NemotronDiar::open(path, latency_s)` then `diarize(&audio, 16_000, threshold)` | `onnx` |
| Per-speaker activity at 10 ms | `NemotronDiar::probabilities` | `onnx` |
| Timed words and tags | `NemotronAsr::open(path)` then `transcribe(&audio, 16_000)` | `onnx` |
| The pure parts | `mel`, `cache`, `schedule`, `segments`, `asr`, `onnx_meta`, `accel` | none |

```toml
[dependencies]
vocal-helper-rs = { version = "0.1", features = ["onnx"] }
```

Without `onnx` the crate builds on Rust 1.82 with no native dependency. With
it, `ort` 2.0.0-rc.15 needs Rust 1.92 and downloads a prebuilt ONNX Runtime at
build time.

## Writing Rust against it

```rust
use std::path::Path;
use vocal_helper_rs::asr_session::NemotronAsr;
use vocal_helper_rs::diar_session::NemotronDiar;

// 16 kHz mono f32 samples.
let mut diar = NemotronDiar::open(Path::new("nemotron-3-diarization.onnx"), Some(1.04))?;
for turn in diar.diarize(&samples, 16_000, 0.5)? {
    println!("{:.2}\t{:.2}\tSPK{}", turn.start, turn.end, turn.speaker);
}

let mut asr = NemotronAsr::open(Path::new("nemotron-asr-ctc.int8.onnx"))?;
let transcript = asr.transcribe(&samples, 16_000)?;
println!("{}", transcript.text());
for w in &transcript.words {
    println!("{:.2}\t{:.2}\t{}", w.start_s, w.end_s, w.text);
}
```

`latency_s` picks the longest streaming schedule that fits that budget (`None`
takes everything the graph allows; 1.04 s is the published low-latency
schedule). `diarize` thresholds the 10 ms probabilities and bridges a speaker's
gaps under 0.5 s.

Where the graphs come from: the diarization graph is
`nemotron3/nemotron-3-diarization.onnx` from the weight bundle the Python twin
downloads; the transcription graph is the one the Python twin caches as
`nemotron-onnx/nemotron-asr-ctc.int8.onnx`, whose metadata must declare the
`nemotron-asr-ctc-1` contract. This crate never downloads anything.

## Four things that look like bugs and are not

**Any rate other than 16 kHz is an error**, not a silent resample. Resample
upstream (`audio-helper-rs`).

**Overlapping turns** for two people talking at once are intended.

**A transcript about 4 % off the Python twin** is the int8 graph's own noise
floor (the Python twin against itself with graph optimizations off is 4.4 %
apart too), not the Rust host: the log-mel matches to 1.3e-6.

**The sessions run on the CPU.** `accel::choose` picks a provider from timings,
but `NemotronDiar` and `NemotronAsr` do not consult it yet; they open their
graph on ONNX Runtime's default provider. On Apple Silicon the CPU is also the
measured winner (CoreML splits the diarization graph into 316 partitions).

## Choosing this crate over the Python twin

Pick this one for Rust calling code, iOS or Android, or one binary per
platform. Not for speed: 99.8 % of the diarization time is in ONNX Runtime's
C++, which `ort` calls too. Pick the Python `vocal-helper` for its CLI, HTTP
API, MCP tools, live microphone, URL input, language identification, summaries,
the other engines and the paid paths.

## References

- `references/triggers.md`: the full catalogue, mirroring the root `TRIGGERS.md`.
- `references/api-reference.md`: every public module, type and function.
- `references/surfaces.md`: which surfaces exist here, and which do not.
