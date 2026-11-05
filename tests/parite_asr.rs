//! L'hôte ASR Rust produit-il les mêmes nombres que le Python ?
//!
//! Les deux chemins alimentent le même graphe `nemotron-asr-ctc` et lisent ses
//! sorties. La référence vient de `vocal_helper.nemotron_asr`, lui-même vérifié
//! contre le front-end NeMo (écart de mel ≤ 1,3e-5, argmax identique sur 750
//! trames). Les fixtures sont écrites par `scripts/make_asr_fixture.py` à partir
//! du vrai graphe : ses métadonnées réelles ainsi que ses sorties sur 30 s d'AMI.

use std::path::Path;

use vocal_helper_rs::asr::{
    greedy_ctc, log_mel, normalize_per_feature, tokens_to_words, AsrSettings, Tag, Word,
};
use vocal_helper_rs::onnx_meta::graph_metadata;

#[derive(serde::Deserialize)]
struct MelCase {
    audio: Vec<f32>,
    valid: usize,
    frames: Vec<Vec<f32>>,
}

#[derive(serde::Deserialize)]
struct CtcCase {
    ids: Vec<i32>,
    logprob: Vec<f32>,
    words: Vec<Word>,
    tags: Vec<Tag>,
}

#[derive(serde::Deserialize)]
struct Reference {
    mel: MelCase,
    ctc: CtcCase,
}

fn settings() -> AsrSettings {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/asr_meta.onnx");
    AsrSettings::from_metadata(&graph_metadata(&path).expect("métadonnées lisibles"))
        .expect("contrat respecté")
}

fn reference() -> Reference {
    serde_json::from_str(include_str!("fixtures/asr_reference.json")).expect("référence lisible")
}

#[test]
fn les_metadonnees_reelles_se_lisent() {
    let s = settings();
    assert_eq!(s.n_mels, 128);
    assert_eq!(s.fb.len(), 128 * 257);
    assert_eq!(s.window.len(), 400);
    assert_eq!(s.vocab.len(), 23862);
    assert_eq!(s.blank_id, 23862);
    assert_eq!(s.mel_per_frame(), 4);
}

#[test]
fn le_mel_normalise_est_celui_du_python() {
    let s = settings();
    let case = reference().mel;
    let (mut mel, valid) = log_mel(&case.audio, 16_000, &s).expect("16 kHz");
    assert_eq!(valid, case.valid);
    normalize_per_feature(&mut mel, valid);
    assert_eq!(mel.len(), case.frames.len());
    let mut worst = 0.0f32;
    for (mine, theirs) in mel.iter().zip(case.frames.iter()) {
        assert_eq!(mine.len(), theirs.len());
        for (&a, &b) in mine.iter().zip(theirs.iter()) {
            worst = worst.max((a - b).abs());
        }
    }
    // Valeurs réduites (écart-type 1) arrondies à 1e-5 : 1e-3 est du bruit de
    // transformée, une erreur de convention se compterait en dixièmes.
    assert!(worst < 1e-3, "écart maximal {worst}");
}

#[test]
fn les_mots_et_les_balises_sont_ceux_du_python() {
    let s = settings();
    let case = reference().ctc;
    let tokens = greedy_ctc(&case.ids, &case.logprob, s.blank_id);
    let (words, tags) = tokens_to_words(&tokens, &s.vocab, s.frame_s);
    assert_eq!(tags, case.tags);
    assert_eq!(words.len(), case.words.len());
    for (mine, theirs) in words.iter().zip(case.words.iter()) {
        assert_eq!(mine.text, theirs.text);
        assert_eq!((mine.start_s, mine.end_s), (theirs.start_s, theirs.end_s));
        assert!((mine.confidence - theirs.confidence).abs() < 1e-4);
    }
}
