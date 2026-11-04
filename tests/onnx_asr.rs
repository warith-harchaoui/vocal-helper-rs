//! Le graphe ASR réel, piloté depuis Rust (feature `onnx`).
//!
//! Ne tourne que si `VH_NEMOTRON_ASR_GRAPH` désigne le graphe : il pèse des
//! centaines de mégaoctets et n'est pas distribué avec la caisse.

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use vocal_helper_rs::asr_session::NemotronAsr;

#[test]
fn le_vrai_graphe_transcrit_une_voix_synthetique_sans_planter() {
    let Some(graph) = std::env::var_os("VH_NEMOTRON_ASR_GRAPH").map(PathBuf::from) else {
        eprintln!("VH_NEMOTRON_ASR_GRAPH non défini : test sauté");
        return;
    };
    let mut asr = NemotronAsr::open(&graph).expect("graphe ouvert");
    // 35 s : deux fenêtres, dont une partielle.
    let audio: Vec<f32> = (0..16_000 * 35)
        .map(|i| 0.05 * ((i as f32) * 0.07).sin())
        .collect();
    let (ids, best, blank) = asr.frames(&audio, 16_000).expect("inférence");
    assert_eq!(ids.len(), (audio.len() / 160).div_ceil(4));
    assert_eq!(best.len(), ids.len());
    assert_eq!(blank.len(), ids.len());
    assert!(asr.frames(&audio, 8_000).is_err());
}
