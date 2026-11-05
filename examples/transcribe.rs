//! Transcrire un WAV 16 kHz mono avec le graphe Nemotron ASR.
//!
//! ```sh
//! cargo run --release --features onnx --example transcribe -- <graphe.onnx> <audio.wav>
//! # sur Ubuntu + NVIDIA : --features cuda ; VOCAL_HELPER_PROVIDER=cuda pour épingler
//! ```

use std::path::Path;
use std::time::Instant;

use vocal_helper_rs::asr_session::NemotronAsr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage : transcribe <graphe.onnx> <audio.wav>");
        std::process::exit(2);
    }
    let mut reader = hound::WavReader::open(&args[2])?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();

    let t0 = Instant::now();
    let mut asr = NemotronAsr::open(Path::new(&args[1]))?;
    let opened = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let transcript = asr.transcribe(&mono, spec.sample_rate)?;
    let elapsed = t1.elapsed().as_secs_f64();
    let seconds = mono.len() as f64 / f64::from(spec.sample_rate);
    println!("{}", transcript.text());
    eprintln!(
        "{} mots, {} balises ; moteurs {:?} ; ouverture {opened:.1} s, {seconds:.1} s d'audio en {elapsed:.2} s (RTF {:.3})",
        transcript.words.len(),
        transcript.tags.len(),
        asr.providers(),
        elapsed / seconds
    );
    Ok(())
}
