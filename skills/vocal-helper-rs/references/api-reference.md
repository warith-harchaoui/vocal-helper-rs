# API reference: `vocal-helper-rs`

`cargo add vocal-helper-rs` (add `--features onnx` to drive the graphs). The
crate is imported as `vocal_helper_rs`. It has no binary: the library is the
whole surface. This page is the map; `cargo doc --features onnx --open` has
every signature. docs.rs builds without `onnx`, so the two sessions are missing
there.

## Features

| Feature | Default | What it adds | Rust floor |
|---------|---------|--------------|-----------|
| none | yes | every pure module below, no native dependency | 1.82 |
| `onnx` | no | `diar_session`, `asr_session`, through `ort` 2.0.0-rc.15 (pinned exactly); downloads a prebuilt ONNX Runtime at build time | 1.92 |

## Sessions (feature `onnx`)

### `diar_session::NemotronDiar`

- `open(path: &Path, latency_s: Option<f64>) -> Result<Self, DiarError>`:
  reads the streaming schedule and the learned silence embedding from the
  graph's metadata, then keeps the longest schedule that fits `latency_s`
  (`None`: everything the graph allows). The offline schedule of the metadata
  is always replaced, as in the Python twin.
- `schedule() -> &Schedule`: the schedule in effect.
- `probabilities(&mut self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>, DiarError>`:
  per-speaker activity every 10 ms, row-major, `num_speakers` values per frame.
- `diarize(&mut self, audio, sample_rate, threshold: f32) -> Result<Vec<Segment>, DiarError>`:
  turns at `threshold`, a speaker's gaps under 0.5 s bridged.
- `DiarError`: `Meta`, `Mel` (wrong rate), `Graph` (no silence embedding,
  impossible schedule), `Ort`.

### `asr_session::NemotronAsr`

- `open(path: &Path) -> Result<Self, SessionError>`: settings and vocabulary
  come from the graph's metadata (contract `nemotron-asr-ctc-1`).
- `open_with(path, providers: Option<&[Provider]>)`: the same, on these
  providers in this order (see below).
- `settings() -> &AsrSettings`, `path() -> &Path`, `providers() -> &[Provider]`.
- `frames(&mut self, audio, sample_rate) -> Result<Frames, SessionError>`: per
  40 ms frame, the token, its log-probability and the blank's
  (`Frames = (Vec<i32>, Vec<f32>, Vec<f32>)`), over independent 30 s windows.
- `transcribe(&mut self, audio, sample_rate) -> Result<Transcript, SessionError>`.
- `Transcript { words: Vec<Word>, tags: Vec<Tag>, speech: Vec<f32> }`, with
  `text()` joining the words by spaces; `speech` is the speech probability
  every 40 ms.
- `SessionError`: `Meta`, `Asr` (metadata out of contract), `Mel`, `Ort`,
  `Accel` (unknown name in `VOCAL_HELPER_PROVIDER`).

`NemotronDiar` has the same `open_with(path, latency_s, providers)` and
`providers()`; `DiarError` gains `Accel` too. Both decide the provider in this
order: the `providers` argument, then `VOCAL_HELPER_PROVIDER` (one name or a
comma-separated order), then a measurement on the device when more than one
provider is compiled in (5 s warm-up, then 30 s timed, verdict kept per graph
digest in `$VOCAL_HELPER_RS_CACHE`, `$XDG_CACHE_HOME/vocal-helper-rs` or
`~/.cache/vocal-helper-rs`). A provider that does not register is skipped with
a warning on stderr; the CPU always closes the list.

### `runtime` (feature `onnx`)

`open_session(&Environment, path, order: &[Provider]) -> ort::Result<Opened>`;
`Opened { session, providers, warnings }`. Registers the providers in order,
skips the ones that fail, rebuilds on the CPU alone if the session cannot be
committed on them. Fails only if the CPU cannot open the graph.

## Pure modules (no feature)

### `mel`: the diarization front end

`log_mel(audio, sample_rate, &MelSettings) -> Result<Vec<Vec<f32>>, MelError>`
(re-exported at the root), `MelSettings` (16 kHz, 512-point transform,
400-sample centred window, 160 hop, 0.97 pre-emphasis, 128 bands, `2^-24` log
floor by default), `mel_filterbank`, `centred_window`.
`MelError::SampleRate { expected, received }` for any rate other than 16 kHz.

### `cache`: speaker cache and FIFO

`StreamingState { spkcache, fifo, cache_seen }` with `new`, `spkcache_flat`,
`fifo_flat`; `Slot { embedding, predictions }`; `streaming_update` (one chunk
into the FIFO, the FIFO into the cache, compressed when it overflows);
`compress_cache`, `log_pred_scores`, `disable_low_scores`, `boost_top_scores`.
All re-exported at the root.

### `schedule`: the streaming schedule

`Schedule` (`Default` = the current checkpoint) with `from_metadata`,
`within(latency_s)`, `named(mode)`, `with_chunk`, `latency_s`, `fits`.
Constants: `DEFAULT_LATENCY_S` (1.04), `MAX_SEQUENCE_FRAMES` (625),
`STREAMING_MODES` (`low_latency` 9+4, `very_low_latency` 6+2,
`ultra_low_latency` 3+1, in 80 ms frames), `META_PREFIX`.

### `segments`: probabilities to turns

`segments_from_probabilities(probabilities, num_speakers, frame_s, threshold, bridge_s) -> Vec<Segment>`;
`Segment { start, end, speaker }` with `duration()`. Speakers are numbered by
first arrival; overlapping turns are kept.

### `asr`: the Nemotron 3.5 host

`AsrSettings::from_metadata`, `mel_per_frame`; `log_mel` (NeMo front end,
constant padding), `normalize_per_feature`, `plan_windows` → `Window`,
`greedy_ctc` → `CtcToken`, `tokens_to_words` → (`Word { text, start_s, end_s,
confidence }`, `Tag { at_s, tag }`), `speech_probability`. `AsrError`:
`Missing`, `Invalid`, `Contract`, `Meta`. Constants `META_PREFIX`, `CONTRACT`,
`WORD_START`, `STD_FLOOR`.

### `onnx_meta`: metadata without the weights

`graph_metadata(path)`, `read_metadata(stream)` (reads `metadata_props` straight
from the protobuf), `base64_decode`, `base64_f32le`. `MetaError`: `Io`, `Format`.

### `accel`: the provider policy

Feature flags, each implying `onnx`: `cuda` (CUDA 13 + cuDNN 9, Linux and
Windows), `tensorrt` (adds TensorRT 10), `coreml` (Apple), `directml`
(Windows), `openvino` (Intel, not together with `cuda`).

`Provider` (`Cpu`, `CoreMl`, `Cuda`, `TensorRt`, `DirectMl`, `Nnapi`, `Qnn`,
`OpenVino`) with `as_str` and `candidates()` (the order to try on this
platform); `Timing { provider, real_time_factor }`; `Verdict { chosen, timings,
graph_digest }` with `real_time_factor` and `applies_to(digest)`;
`choose(&[Timing]) -> Option<Provider>` (the fastest measured, `None` without
measurements); `holds_live(stage_rtf, other_rtf_same_unit)`; `PROBE_AUDIO`
(30 s), `PROBE_WARMUP` (5 s).
`Provider::from_name` (the Python twin's aliases), `feature`, `is_compiled`,
`compiled()`; `PROVIDER_ENV`, `CACHE_ENV`; `requested_order`, `parse_order`
(`AccelError::UnknownName`); the `Registrar` trait and
`register_in_order(order, &mut registrar) -> Registration { active, warnings }`;
`measure(candidates, audio_s, open, time)`; `graph_digest(path)` (FNV-1a over
the size, the first and the last MiB); `verdict_dir`, `load_verdict`,
`save_verdict`; `Verdict` also carries `tried` and `still_holds(digest,
candidates)`. All pure: nothing in `accel` opens a session.

## Examples (feature `onnx`)

```sh
cargo run --release --features onnx --example diarize -- <graph.onnx> <clip.wav> [--probs <out.f32>]
cargo run --release --features onnx --example transcribe -- <graph.onnx> <clip.wav>
```

Both read a 16 kHz WAV and average its channels to mono. `diarize` prints
`start\tend\tSPKn` lines and the RTF on stderr; `--probs` writes the 10 ms
probabilities as little-endian `f32`. `transcribe` prints the transcript.
Exit code 2 on a wrong argument count.
