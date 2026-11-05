# TRIGGERS: vocal-helper-rs

The exhaustive catalogue of what `vocal-helper-rs` does and what should reach
for it. The Python twin, [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper),
has its own catalogue. The two overlap; the section on choosing between them
is the part that matters.

`vocal-helper-rs` runs the **free local Nemotron paths** of `vocal-helper` from
Rust, with no Python and no torch: streaming **diarization** with Nemotron 3
Diarization and **transcription** with Nemotron 3.5, each through its ONNX
graph, driven by [`ort`](https://crates.io/crates/ort) behind the `onnx`
feature. It is a library: there is no binary, no HTTP API and no MCP server.
The graphs are not shipped with the crate; the caller passes their path.

## What it does → how to invoke

| Intent | Library | Feature |
|--------|---------|---------|
| Who spoke when (speech turns) | `diar_session::NemotronDiar::open` + `diarize` | `onnx` |
| Per-speaker activity every 10 ms | `NemotronDiar::probabilities` | `onnx` |
| Timed words, speaker / event tags, text | `asr_session::NemotronAsr::open` + `transcribe` | `onnx` |
| Per-frame CTC tokens (every 40 ms) | `NemotronAsr::frames` | `onnx` |
| Log-mel frames the diarization graph expects | `log_mel`, `MelSettings` | none |
| Speaker cache and FIFO between streaming steps | `streaming_update`, `StreamingState`, `compress_cache` | none |
| Streaming schedule within a latency budget | `Schedule::within`, `Schedule::named`, `STREAMING_MODES` | none |
| Probabilities to speech turns | `segments_from_probabilities`, `Segment` | none |
| Greedy CTC decoding, words, tags | `asr::greedy_ctc`, `asr::tokens_to_words` | none |
| Read a graph's `metadata_props` without its weights | `onnx_meta::graph_metadata` | none |
| Pick an execution provider from on-device timings | `choose`, `holds_live`, `Verdict` | none |
| Run on CUDA, TensorRT, CoreML, DirectML or OpenVINO, CPU fallback | `open_with(.., Some(&[Provider::Cuda]))`, `VOCAL_HELPER_PROVIDER=cuda`, `providers()` | `cuda`, `tensorrt`, `coreml`, `directml`, `openvino` |

Two runnable examples, both behind `onnx`, read a 16 kHz WAV (several channels
are averaged to mono); `--probs` also writes the raw probabilities as
little-endian `f32`:
`cargo run --release --features onnx --example diarize -- <graph.onnx> <clip.wav> [--probs <out.f32>]`
and `cargo run --release --features onnx --example transcribe -- <graph.onnx> <clip.wav>`.

## Natural-language phrasings that should fire

- **Diarize**: "diarize this in Rust", "who spoke when, without Python",
  "speaker turns on device", "streaming speaker diarization in Rust",
  "Nemotron 3 Diarization from Rust", "Sortformer ONNX in Rust".
- **Transcribe**: "transcribe in Rust", "speech to text without Python",
  "Nemotron 3.5 from Rust", "CTC transcription with timed words in Rust".
- **Embed**: "diarization on iOS / Android", "speech in a single binary",
  "no interpreter, no torch", "run the vocal-helper graphs from Rust".
- **The parts**: "NeMo log-mel in Rust", "speaker cache / FIFO bookkeeping",
  "streaming schedule for a latency budget", "greedy CTC decoder in Rust",
  "read ONNX metadata_props without loading the weights", "choose an ONNX
  Runtime execution provider by measuring".
- **Accelerate**: "diarize on an NVIDIA GPU from Rust", "Rust speech on CUDA /
  TensorRT", "Ubuntu + NVIDIA, which CUDA for vocal-helper-rs", "CoreML from
  Rust for speech".
- **The crate itself**: `cargo add vocal-helper-rs`, `vocal_helper_rs`,
  "the Rust vocal helper", "the Rust port of vocal-helper".

## Things that look like failures and are not

- **Any sample rate other than 16 kHz is an error** (`MelError::SampleRate`),
  not a silent resample: an implicit resample would give plausible, wrong frames.
- **Two overlapping turns** for two speakers talking at once are intended: the
  model can say so. Flattening it would lose the information.
- **A transcript a few percent off the Python twin** is the int8 graph's noise
  floor, not the Rust host: 4.4 % WER over 20 min of AMI, the same as the Python
  twin against itself with graph optimizations off.
- **A warning on stderr and a session on the CPU** after asking for CUDA on a
  machine without a GPU, or with CUDA 12 where the prebuilt runtime wants
  CUDA 13, is the intended fallback, not a failure: `providers()` says what was
  kept. Only an unknown name in `VOCAL_HELPER_PROVIDER` is an error.
- **A first open that takes a minute** with several providers compiled in is
  the one-off measurement; the verdict is kept per graph and per machine.

## Choosing between the Rust crate and the Python twin

Reach for **`vocal-helper-rs`** when the calling code is Rust, when Python
cannot go there (iOS, Android), or when one binary per platform is easier to
ship than an interpreter and a scientific stack. Not for speed: 99.8 % of the
diarization time is already in ONNX Runtime's C++, which `ort` calls too.

Reach for the **Python `vocal-helper`** for everything else it does: the CLI,
the HTTP API and MCP tools, the transcript viewer, live microphone sessions, URL
input, language identification, the rolling summary, the whisper / pyannote /
NeMo / sherpa engines, the paid paths (Gladia, pyannoteAI), automatic weight
download.

## When NOT to use vocal-helper-rs (SKIP)

- A command line, an HTTP API or MCP tools → the Python `vocal-helper`.
- Paid or hosted transcription and diarization (Gladia, pyannoteAI) → the
  Python `vocal-helper`; they are not written here.
- Downloading the model graphs → the Python `vocal-helper` fetches them; this
  crate only reads a path.
- Capturing a live microphone → `capture-helper-rs`.
- Decoding, resampling or converting audio files → `audio-helper-rs`.
- Downloading media from a URL → `youtube-helper-rs` or `podcast-helper-rs`.
- Speech synthesis or voice cloning. This crate listens; it does not speak.

## See also

- [`README.md`](README.md) / [`LISEZMOI.md`](LISEZMOI.md): scope, measurements and install.
- [`CHANGELOG.md`](CHANGELOG.md): what changed, version by version.
