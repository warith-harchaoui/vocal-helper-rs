# Vocal Helper (Rust)

[🇫🇷](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/LISEZMOI.md) · [🇬🇧](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/README.md)

[![crates.io](https://img.shields.io/crates/v/vocal-helper-rs.svg)](https://crates.io/crates/vocal-helper-rs) [![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE)

Rust rewrite of [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper)'s free local diarization path — Nemotron 3 Diarization through its ONNX graph. Not a line-by-line port, idiomatic Rust throughout.

## Why this exists, stated honestly

**Not for speed.** That was measured, not assumed. Profiling the diarization path on two minutes
of audio, with the graph call and the feature front end timed separately:

| | time | share |
|---|---:|---:|
| ONNX graph *(C++, identical when called from Rust)* | 90.47 s | **99.8 %** |
| log-mel front end *(numpy, FFT in C)* | 0.06 s | 0.1 % |
| cache bookkeeping + Python | 0.10 s | 0.1 % |

A Rust port's ceiling is **0.2 % of the wall time** — a real-time factor of 0.754 instead of
0.755. The `ort` crate calls the same C++ library, and the Python pieces are either vectorised or
working on 264×512 arrays about once a second. Anyone promising an RTF gain here has not measured.

**For places to run.** On iOS and Android the Python path does not exist: one does not reasonably
embed CPython and a scientific stack in a mobile application. That is a *possibility* argument, not
a performance one, and it is decisive. Add to it one binary per platform, no dependency resolution
at install time, no interpreter to start, and a small memory footprint for a background listener.

Same engine, same numbers, same speed — somewhere Python does not reach.

## Scope

Everything that lives **around** the graph is here, and that is the bulk of the work. What is not
written yet is named as such rather than implied.

| module | what it does | state |
|---|---|---|
| `mel` | the log-mel front end the graph expects, to the frame | **done**, parity with Python proven |
| `cache` | speaker cache and FIFO bookkeeping across streaming steps | **done**, parity proven |
| `schedule` | the streaming schedule and the two ceilings that bound it | **done**, parity proven |
| `segments` | per-frame probabilities to speech turns | **done**, parity proven |
| `accel` | choosing an execution provider from measurements taken on the device | **policy done**, tested; not yet applied by the sessions, which run on ONNX Runtime's CPU default |
| `asr` | Nemotron 3.5 transcription host: NeMo front end, windows, greedy CTC, timed words, speaker/event tags | **done**, parity proven on the real graph's metadata and outputs |
| `onnx_meta` | reading a graph's `metadata_props` without loading its weights | **done** |
| ONNX Runtime binding | driving the graph itself, through `ort`, behind the `onnx` feature | **transcription done** (`asr_session`, `examples/transcribe.rs`; 0.86 % WER from the Python twin on int8, RTF 0.15 on CPU); **diarization done** (`diar_session`, `examples/diarize.rs`; probabilities within 1.7e-6 of the Python twin over 3 min of AMI, cache compression included; 1.9e-6 on another minute of AMI, re-measured for this release) |
| paid paths | Gladia and pyannoteAI, batch and live | not written |

```sh
cargo test                                  # 87 tests, parity included
cargo clippy --all-targets -- -D warnings
```

## Installation and use

```toml
[dependencies]
vocal-helper-rs = { version = "0.1", features = ["onnx"] }
```

Without `onnx` the crate is the pure part (front ends, cache, schedule, segments, CTC decoding) and
builds on Rust 1.82 with no native dependency. With `onnx` it pulls `ort` 2.0.0-rc.15, which needs
Rust 1.92 and downloads a prebuilt ONNX Runtime at build time.

**The graphs are not shipped with the crate**, they weigh hundreds of megabytes: you pass their path.
The diarization graph is the single-file ONNX export of
[nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) made for
[Vernacula](https://github.com/christopherthompson81/vernacula), whose metadata carries the streaming
schedule and the silence embedding; it is the `nemotron3/nemotron-3-diarization.onnx` file of the
weight bundle the Python twin `vocal-helper` downloads.

```rust
use std::path::Path;
use vocal_helper_rs::diar_session::NemotronDiar;

// 16 kHz mono f32 samples; any other rate is an error, not a silent resample.
let mut diar = NemotronDiar::open(Path::new("nemotron-3-diarization.onnx"), Some(1.04))?;
for turn in diar.diarize(&samples, 16_000, 0.5)? {
    println!("{:.2}\t{:.2}\tSPK{}", turn.start, turn.end, turn.speaker);
}
```

`examples/diarize.rs` and `examples/transcribe.rs` do the same from a WAV file:

```sh
cargo run --release --features onnx --example diarize -- nemotron-3-diarization.onnx clip.wav
cargo run --release --features onnx --example transcribe -- <asr-graph.onnx> clip.wav
```

## A port is only worth its parity

The graph does not take audio, it takes log-mel frames. Its configuration gives the skeleton —
16 kHz, 400-sample window, 160 hop, 512-point transform, 0.97 pre-emphasis, 128 bands — but not
the details that decide the result. Those were **recovered by comparing against the model's own
extractor**, not guessed, and two of them matter:

- The 400-point analysis window is **centred** in the 512-point transform, not flush at the start.
  Putting it at the start multiplies the gap against the official outputs sixteenfold — 0.530
  instead of 0.033 — because the phase shift changes the spectrum.
- The log floor is `2^-24`, read off the minimum of the official outputs: −16.36, which is exactly
  its logarithm.

Both paths feed the **same graph**, so if they disagree the mobile path silently produces different
probabilities and no unit test on either side would notice. `tests/parite_mel.rs` therefore
compares against a fixture generated by `vocal_helper.nemotron.log_mel` on four signals chosen for
what can diverge — rich noise, a pure tone on the Slaney hinge, two distant tones to catch leakage
between neighbouring filters, and silence, where the floor must be reached exactly. Agreement is
within `1e-3`.

That makes a chain of two links: official extractor → Python (0.006 mean gap) → Rust (≤ 1e-3).

## Acceleration: measure the device, do not trust the reasoning

The obvious argument about accelerators is wrong, and it cost us. ONNX Runtime has no Metal
provider, so on Apple Silicon acceleration goes through CoreML, and putting it first looked
self-evident. Measured on the diarization graph, CoreML gives a real-time factor of **1.257** —
slower than real time, therefore unusable — against **0.528** for CPU alone. The session log says
why: CoreML takes 1193 of 2827 nodes and splits them into **316 partitions**, so every streaming
step pays that many crossings between two engines.

The second "obvious" setting was wrong too: forcing eight threads on an eight-performance-core
machine measures 0.953 against 0.528 when the runtime chooses.

The lesson is not that CoreML is bad — it will be excellent on another graph, or in a release that
partitions better. The lesson is that a hardcoded order will be wrong somewhere, so the policy in
`accel` chooses from **timings taken on the device** (`choose`), and a `Verdict` is tied to a graph
digest (`applies_to`), because partitioning depends on the operators and therefore on the model:
that is precisely the mistake a hardcoded order makes.

The policy is pure — no model, no network, no ONNX — so it is entirely testable.

**What this release does not do yet**: the sessions do not consult that policy. `NemotronDiar` and
`NemotronAsr` open their graph on ONNX Runtime's default provider, the CPU — which is also the
measured winner on Apple Silicon above. Timing the providers on the device, the explicit ladder the
Python twin has (a provider named by the caller, then `VOCAL_HELPER_PROVIDER`, then the measured
verdict) and the error on an unknown name are the next step, not a feature of this crate today.


## What the numbers above come from

The measurement campaign behind these choices is written up in vocal-helper's
[`doc/diarization-results.en.md`](https://github.com/warith-harchaoui/vocal-helper/blob/main/doc/diarization-results.en.md),
including the dead ends: four threshold parameterisations refuted, centroid-free clustering closed,
and the discovery that AMI's reference cannot arbitrate because it is transcribed from headset
microphones while the engine is given the room mix.

## Related

Part of the same author's local-first tooling as [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper) (Python), [`audio-helper-rs`](https://github.com/warith-harchaoui/audio-helper-rs), [`capture-helper-rs`](https://github.com/warith-harchaoui/capture-helper-rs), [`podcast-helper-rs`](https://github.com/warith-harchaoui/podcast-helper-rs), [`youtube-helper-rs`](https://github.com/warith-harchaoui/youtube-helper-rs), and the [AI Helpers](https://github.com/warith-harchaoui/ai-helpers) suite. Independent rewrite, not a binding.

## License

BSD-3-Clause, see [LICENSE](LICENSE). The model weights it loads are not ours and carry their own
terms — Nemotron 3 Diarization is NVIDIA's, under the OpenMDW License 1.1.

## Author

[Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui)
