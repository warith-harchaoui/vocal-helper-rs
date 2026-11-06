# Vocal Helper (Rust)

[🇫🇷](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/LISEZMOI.md) · [🇬🇧](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/README.md)

[![crates.io](https://img.shields.io/crates/v/vocal-helper-rs.svg)](https://crates.io/crates/vocal-helper-rs) [![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE)

Rust rewrite of [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper)'s free local Nemotron paths: streaming diarization with Nemotron 3 Diarization and transcription with Nemotron 3.5, each through its ONNX graph. Not a line-by-line port, idiomatic Rust throughout.

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
| `accel` | choosing an execution provider from measurements taken on the device | **done** and applied by both sessions: caller, then `VOCAL_HELPER_PROVIDER`, then an on-device measurement; CPU fallback with a warning; features `cuda`, `tensorrt`, `coreml`, `directml`, `openvino` |
| `asr` | Nemotron 3.5 transcription host: NeMo front end, windows, greedy CTC, timed words, speaker/event tags | **done**, parity proven on the real graph's metadata and outputs |
| `onnx_meta` | reading a graph's `metadata_props` without loading its weights | **done** |
| ONNX Runtime binding | driving the graph itself, through `ort`, behind the `onnx` feature | **transcription done** (`asr_session`, `examples/transcribe.rs`; 4.4 % WER from the Python twin on int8 over 20 min of AMI, which is the int8 graph's own noise floor (the Python twin against itself with graph optimizations off: 4.4 %); same mel to 1.3e-6; RTF 0.05 on CPU against 0.022 to 0.028 for the Python twin; measured 2026-11-05 on an Apple M2 Max, `tests/jumeau_ami.rs`); **diarization done** (`diar_session`, `examples/diarize.rs`; probabilities within 1.7e-6 of the Python twin over 3 min of AMI, cache compression included; 1.9e-6 on another minute of AMI, re-measured for this release) |
| paid paths | Gladia and pyannoteAI, batch and live | not written |

```sh
cargo test                                  # 116 tests, parity included
cargo clippy --all-targets -- -D warnings
```

The end-to-end check against the Python twin needs the int8 graph and the AMI subset that
`vocal-helper` caches, so it is ignored by default. One command re-runs it:

```sh
VH_NEMOTRON_ASR_GRAPH=~/.cache/vocal-helper/nemotron-onnx/nemotron-asr-ctc.int8.onnx \
VH_AMI_SUBSET=~/.cache/vocal-helper/ami-subset \
cargo test --release --features onnx --test jumeau_ami -- --ignored --nocapture
```

It transcribes 20 one-minute AMI clips and fails above 6 % WER from the Python transcripts
kept in `tests/fixtures/ami_jumeau_int8.json` (or above 15 % on any one clip). The gap is not
zero and should not be: two ONNX Runtime builds round the int8 kernels differently, and the
frames where two tokens nearly tie flip. `scripts/make_ami_jumeau.py` rewrites the fixture.

## Installation and use

```toml
[dependencies]
vocal-helper-rs = { version = "0.2", features = ["onnx"] }
```

Without `onnx` the crate is the pure part (front ends, cache, schedule, segments, CTC decoding) and
builds on Rust 1.82 with no native dependency. With `onnx` it pulls `ort` 2.0.0-rc.15, which needs
Rust 1.92 and downloads a prebuilt ONNX Runtime at build time.

One feature per execution provider, each implying `onnx`, each tied to the prebuilt runtimes `ort`
can download, so a feature built off its platform fails in `ort-sys`'s build script, not at run
time:

| feature | provider | platforms with a prebuilt runtime |
|---|---|---|
| `onnx` | CPU only | all |
| `cuda` | CUDA 13 + cuDNN 9 | Linux x86_64 and aarch64, Windows x86_64 |
| `tensorrt` | TensorRT 10, then CUDA | same as `cuda` (same download) |
| `coreml` | CoreML | macOS, iOS |
| `directml` | DirectML | Windows |
| `openvino` | OpenVINO | Linux x86_64, Windows x86_64; not together with `cuda` |

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

Both expect 16 kHz and average several channels to mono. `diarize` prints one turn per line and,
given `--probs <out.f32>`, also writes the 10 ms probabilities as little-endian `f32`.

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

Both sessions apply it, in the Python twin's order of authority:

1. the `providers` argument of `NemotronAsr::open_with` / `NemotronDiar::open_with`;
2. else `VOCAL_HELPER_PROVIDER`: one name (`cuda`, `tensorrt`, `coreml`, `directml`, `openvino`,
   `cpu`…) or an order (`tensorrt,cuda`). An unknown name is an error, so a typo cannot pass for
   CUDA while the CPU runs;
3. else, when more than one provider is compiled in, a **measurement on the device**: each one
   gets a 5 s warm-up then 30 s of synthetic audio timed, the fastest wins, and the `Verdict` is
   kept under the graph's digest in `$VOCAL_HELPER_RS_CACHE` (default
   `~/.cache/vocal-helper-rs`). With `onnx` alone only the CPU is compiled and nothing is measured.

The measurement is paid once per graph and per machine, and it is not free: on an M2 Max with
`coreml`, the first open of the diarization graph took about a minute, and it found again what is
written above (CPU 0.74 against CoreML 1.12 on that run); for the ASR graph the two were level
(0.084 against 0.086). Pin a provider to skip it.

A provider that fails to register — no GPU, wrong CUDA version, feature not compiled — is skipped
with a warning on stderr and the next one is tried; the CPU always closes the list, and a session
that cannot be built on the providers kept is rebuilt on the CPU alone. Opening never fails for
want of an accelerator. Unlike the Python twin, a known provider missing from the build is skipped
with that warning rather than refused.

## Ubuntu + NVIDIA

```toml
vocal-helper-rs = { version = "0.2", features = ["cuda"] }        # or "tensorrt"
```

`ort` 2.0.0-rc.15 downloads **ONNX Runtime 1.31.0** built by pyke for **CUDA 13 only**: there is no
CUDA 12 build to download (`ORT_CUDA_VERSION=12` makes the build fail rather than fall back). The
CUDA build of ONNX Runtime is statically linked into your binary; the providers stay separate
libraries that `cargo build` links next to it in `target/<profile>/`:
`libonnxruntime_providers_shared.so`, `libonnxruntime_providers_cuda.so` and, for `tensorrt`,
`libonnxruntime_providers_tensorrt.so`. **Ship them next to your executable.**

What the machine needs, read with `ldd` off those libraries in this repository's CI:

| feature | libraries looked up at run time | from |
|---|---|---|
| `cuda` | `libcuda.so.1` | the NVIDIA driver, branch 580 or newer (what CUDA 13 requires) |
| `cuda` | `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libcurand.so.10` | CUDA 13 runtime libraries |
| `cuda` | `libcudnn.so.9` and its sub-libraries | cuDNN 9 built for CUDA 13 (required by `ort`'s preload list, though not a direct link of the CUDA provider) |
| `tensorrt` | `libnvinfer.so.10`, `libnvonnxparser.so.10` | TensorRT 10 for CUDA 13 |

On Ubuntu with NVIDIA's apt repository, `sudo apt install cuda-libraries-13-0 libcudnn9-cuda-13`
covers `cuda` (add `tensorrt` for the TensorRT provider). To check a machine:

```sh
nvidia-smi                                   # driver loaded, "CUDA Version: 13.x" or above
ldconfig -p | grep -E 'libcudart.so.13|libcublas.so.13|libcudnn.so.9'
ldd target/release/libonnxruntime_providers_cuda.so | grep 'not found'   # must print nothing
VOCAL_HELPER_PROVIDER=cuda cargo run --release --features cuda --example transcribe -- graph.onnx clip.wav
```

The last line prints `moteurs [Cuda, Cpu]` when CUDA really runs. **Nothing is lost when it does
not**: a provider that fails to register (no GPU, driver too old, CUDA 12 libraries, cuDNN
missing) is skipped with a line on stderr naming ONNX Runtime's own error, and the session opens
on the CPU. `providers()` on either session says what was kept.

**What is verified, and what is not.** CI builds and tests the crate on `ubuntu-latest` with
`--features cuda,tensorrt`: the CUDA build of ONNX Runtime is downloaded and linked, and every
session falls back to the CPU on that runner, which has no GPU. No free runner has one, so
**inference on an actual NVIDIA GPU has not been run by this repository**. The sonames above come
from the libraries themselves, not from that run.


## What the numbers above come from

The measurement campaign behind these choices is written up in vocal-helper's
[`doc/diarization-results.en.md`](https://github.com/warith-harchaoui/vocal-helper/blob/main/doc/diarization-results.en.md),
including the dead ends: four threshold parameterisations refuted, centroid-free clustering closed,
and the discovery that AMI's reference cannot arbitrate because it is transcribed from headset
microphones while the engine is given the room mix.

## Agent skill

`skills/vocal-helper-rs/` packages this crate as a **Claude Skill** and an **OpenCode skill**, so
an agent can discover it and write Rust against it on the user's behalf. Installation for both hosts
is in [`skills/README.md`](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/skills/README.md).

Its trigger catalogue is enforced, not decorative:
[`TRIGGERS.md`](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/TRIGGERS.md) is the
user-facing catalogue, `references/triggers.md` is its agent-facing mirror. `tests/skill.rs` fails
`cargo test` if the two drift apart, if the skill's `description` exceeds the 1024-character cap or
loses its SKIP clause, or if `SKILL.md` cites a reference that does not exist. The pre-push hook
runs it; `scripts/install-hooks.sh` enables that hook once per clone.

## Related

Part of the same author's local-first tooling as [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper) (Python), [`audio-helper-rs`](https://github.com/warith-harchaoui/audio-helper-rs), [`capture-helper-rs`](https://github.com/warith-harchaoui/capture-helper-rs), [`podcast-helper-rs`](https://github.com/warith-harchaoui/podcast-helper-rs), [`youtube-helper-rs`](https://github.com/warith-harchaoui/youtube-helper-rs), and the [AI Helpers](https://github.com/warith-harchaoui/ai-helpers) suite. Independent rewrite, not a binding.

## License

BSD-3-Clause, see [LICENSE](LICENSE). The model weights it loads are not ours and carry their own
terms — Nemotron 3 Diarization is NVIDIA's, under the OpenMDW License 1.1.

## Author

[Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui)
