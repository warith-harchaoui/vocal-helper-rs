//! Vingt minutes d'AMI transcrites depuis Rust, comparées au jumeau Python.
//!
//! `tests/fixtures/ami_jumeau_int8.json` garde le texte que
//! `vocal_helper.nemotron_asr` tire du graphe int8 sur le CPU (écrit par
//! `scripts/make_ami_jumeau.py`). Ce test transcrit les mêmes minutes avec
//! `asr_session` et mesure le WER de l'un contre l'autre.
//!
//! Le graphe int8 n'est pas déterministe d'un build d'ONNX Runtime à l'autre :
//! les trames où deux jetons sont presque à égalité basculent. Le jumeau Python
//! contre lui-même, optimisations du graphe coupées, s'écarte déjà de 4,4 % de
//! WER sur ces vingt minutes. L'écart Rust se juge contre ce plancher, pas
//! contre zéro.
//!
//! Ignoré par défaut : il lui faut le graphe (669 Mo) et l'audio, qui ne sont pas
//! distribués avec la caisse.
//!
//! ```sh
//! VH_NEMOTRON_ASR_GRAPH=~/.cache/vocal-helper/nemotron-onnx/nemotron-asr-ctc.int8.onnx \
//! VH_AMI_SUBSET=~/.cache/vocal-helper/ami-subset \
//! cargo test --release --features onnx --test jumeau_ami -- --ignored --nocapture
//! ```

#![cfg(feature = "onnx")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use vocal_helper_rs::asr_session::NemotronAsr;

/// Le WER Rust contre Python sur les vingt minutes ne doit pas dépasser ceci.
/// Mesuré : 4,38 % (2026-11-05, Apple M2 Max), le plancher Python contre
/// Python étant de 4,4 %.
const MAX_CORPUS_WER: f64 = 0.06;

/// Aucune minute ne doit s'écarter de plus que ceci (mesuré : 9,3 % au pire).
const MAX_CLIP_WER: f64 = 0.15;

#[derive(serde::Deserialize)]
struct Clip {
    text: String,
}

#[derive(serde::Deserialize)]
struct Fixture {
    onnxruntime: String,
    clips: BTreeMap<String, Clip>,
}

fn env_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} doit désigner son chemin (voir l'en-tête du test)"))
}

fn read_mono(path: &PathBuf) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("WAV lisible");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / scale)
                .collect()
        }
    };
    let channels = usize::from(spec.channels);
    let mono = samples
        .chunks(channels)
        .map(|f| f.iter().sum::<f32>() / channels as f32)
        .collect();
    (mono, spec.sample_rate)
}

/// Distance d'édition en mots.
fn word_edits(a: &[&str], b: &[&str]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut diag = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let next = (row[j + 1] + 1)
                .min(row[j] + 1)
                .min(diag + usize::from(x != y));
            diag = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

#[test]
fn la_distance_en_mots_est_celle_de_levenshtein() {
    assert_eq!(word_edits(&["a", "b", "c"], &["a", "c"]), 1);
    assert_eq!(word_edits(&["a", "b"], &["a", "x", "b", "y"]), 2);
    assert_eq!(word_edits(&[], &["a"]), 1);
}

#[test]
#[ignore = "il faut le graphe int8 et le sous-corpus AMI de vocal-helper"]
fn vingt_minutes_d_ami_restent_au_plancher_du_jumeau_python() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/ami_jumeau_int8.json")).expect("fixture");
    let graph = env_path("VH_NEMOTRON_ASR_GRAPH");
    let ami = env_path("VH_AMI_SUBSET");
    let mut asr = NemotronAsr::open(&graph).expect("graphe ouvert");

    let (mut edits, mut words, mut audio_s, mut compute_s) = (0usize, 0usize, 0.0f64, 0.0f64);
    let mut worst = (0.0f64, String::new());
    for (i, (id, clip)) in fixture.clips.iter().enumerate() {
        let (audio, sr) = read_mono(&ami.join(id).join("clip.wav"));
        let t0 = Instant::now();
        let text = asr.transcribe(&audio, sr).expect("transcription").text();
        let elapsed = t0.elapsed().as_secs_f64();
        // La première minute paie l'échauffement de la session : hors RTF.
        if i > 0 {
            compute_s += elapsed;
            audio_s += audio.len() as f64 / f64::from(sr);
        }
        let reference: Vec<&str> = clip.text.split_whitespace().collect();
        let ours: Vec<&str> = text.split_whitespace().collect();
        let e = word_edits(&ours, &reference);
        let wer = e as f64 / reference.len().max(1) as f64;
        eprintln!(
            "{id}: {e}/{} mots, WER {:.2} %",
            reference.len(),
            100.0 * wer
        );
        if wer > worst.0 {
            worst = (wer, id.clone());
        }
        edits += e;
        words += reference.len();
    }
    let wer = edits as f64 / words as f64;
    eprintln!(
        "{} minutes : WER {:.2} % contre le jumeau Python (onnxruntime {}), pire {} à {:.2} %, RTF {:.3}",
        fixture.clips.len(),
        100.0 * wer,
        fixture.onnxruntime,
        worst.1,
        100.0 * worst.0,
        compute_s / audio_s
    );
    assert!(wer <= MAX_CORPUS_WER, "WER {wer:.4} > {MAX_CORPUS_WER}");
    assert!(
        worst.0 <= MAX_CLIP_WER,
        "{} : WER {:.4} > {MAX_CLIP_WER}",
        worst.1,
        worst.0
    );
}
