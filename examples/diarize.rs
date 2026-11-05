//! Diariser un WAV 16 kHz mono avec le graphe Nemotron 3 Diarization.
//!
//! ```sh
//! cargo run --release --features onnx --example diarize -- <graphe.onnx> <audio.wav> [--probs <sortie.f32>]
//! ```

use std::path::Path;
use std::time::Instant;

use vocal_helper_rs::diar_session::NemotronDiar;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 && args.len() != 5 {
        eprintln!("usage : diarize <graphe.onnx> <audio.wav> [--probs <sortie.f32>]");
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

    let mut diar = NemotronDiar::open(Path::new(&args[1]), Some(1.04))?;
    let t0 = Instant::now();
    let probabilities = diar.probabilities(&mono, spec.sample_rate)?;
    let elapsed = t0.elapsed().as_secs_f64();
    if args.len() == 5 {
        let bytes: Vec<u8> = probabilities.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&args[4], bytes)?;
    }
    let segments = vocal_helper_rs::segments_from_probabilities(
        &probabilities,
        diar.schedule().num_speakers,
        0.01,
        0.5,
        0.5,
    );
    for s in &segments {
        println!("{:.2}\t{:.2}\tSPK{}", s.start, s.end, s.speaker);
    }
    let seconds = mono.len() as f64 / f64::from(spec.sample_rate);
    eprintln!(
        "{} tours ; moteurs {:?} ; {seconds:.1} s d'audio en {elapsed:.2} s (RTF {:.3})",
        segments.len(),
        diar.providers(),
        elapsed / seconds
    );
    Ok(())
}
