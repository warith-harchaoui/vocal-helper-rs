//! Le choix du moteur d'exécution contre un vrai ONNX Runtime (feature `onnx`).
//!
//! `fixtures/double.onnx` est un graphe d'un seul nœud (`y = 2x`) : il s'ouvre
//! partout, donc le repli se vérifie sans modèle et sans GPU. Sur le runner
//! Ubuntu de la CI, compilé avec `--features cuda`, l'enregistrement de CUDA
//! échoue faute de pilote : c'est exactement le chemin à garder vert.
//! `VH_EXPECT_NO_GPU=1` (posé par la CI) rend l'attente stricte.

#![cfg(feature = "onnx")]

use std::path::{Path, PathBuf};

use ort::value::Tensor;
use vocal_helper_rs::runtime::open_session;
use vocal_helper_rs::Provider;

fn double() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/double.onnx")
}

fn environment() -> ort::environment::Environment {
    ort::init()
        .with_name("vocal-helper-rs-tests")
        .build()
        .expect("environnement")
}

fn sans_gpu_attendu() -> bool {
    std::env::var("VH_EXPECT_NO_GPU").is_ok_and(|v| v == "1")
}

fn tourne(session: &mut ort::session::Session) -> Vec<f32> {
    let x = Tensor::from_array(([3usize], vec![1.0f32, 2.0, 3.0])).unwrap();
    let out = session.run(ort::inputs!["x" => x]).expect("inférence");
    out["y"].try_extract_tensor::<f32>().unwrap().1.to_vec()
}

#[test]
fn demander_cuda_ouvre_toujours_une_session_qui_calcule_juste() {
    let mut opened = open_session(&environment(), &double(), &[Provider::Cuda]).expect("ouverte");
    assert_eq!(opened.providers.last(), Some(&Provider::Cpu));
    if opened.providers[0] != Provider::Cuda {
        // Repli : il doit avoir été dit, pas subi en silence.
        assert_eq!(opened.providers, vec![Provider::Cpu]);
        assert_eq!(opened.warnings.len(), 1, "{:?}", opened.warnings);
        assert!(opened.warnings[0].contains("CUDAExecutionProvider"));
    }
    if sans_gpu_attendu() {
        assert_eq!(
            opened.providers,
            vec![Provider::Cpu],
            "aucun GPU ici, CUDA ne peut pas avoir été retenu"
        );
    }
    assert_eq!(tourne(&mut opened.session), vec![2.0, 4.0, 6.0]);
}

#[cfg(feature = "cuda")]
#[test]
fn compile_avec_cuda_sans_gpu_l_echec_vient_du_runtime_et_non_de_la_compilation() {
    // Avec la feature, le moteur est bien essayé : l'avertissement cite
    // l'erreur d'ONNX Runtime, pas l'absence de feature.
    let opened = open_session(
        &environment(),
        &double(),
        &[Provider::TensorRt, Provider::Cuda],
    )
    .expect("ouverte");
    if sans_gpu_attendu() {
        assert_eq!(opened.providers, vec![Provider::Cpu]);
        for w in &opened.warnings {
            assert!(!w.contains("sans la feature"), "{w}");
            assert!(w.contains("n'a pas pu s'enregistrer"), "{w}");
        }
    }
}

#[test]
fn un_ordre_entierement_indisponible_retombe_sur_le_cpu() {
    let order = [
        Provider::TensorRt,
        Provider::Cuda,
        Provider::DirectMl,
        Provider::OpenVino,
        Provider::Nnapi,
    ];
    let mut opened = open_session(&environment(), &double(), &order).expect("ouverte");
    assert_eq!(opened.providers.last(), Some(&Provider::Cpu));
    assert_eq!(tourne(&mut opened.session), vec![2.0, 4.0, 6.0]);
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
#[test]
fn sur_mac_coreml_s_enregistre_vraiment() {
    let mut opened = open_session(&environment(), &double(), &[Provider::CoreMl]).expect("ouverte");
    assert_eq!(opened.providers, vec![Provider::CoreMl, Provider::Cpu]);
    assert_eq!(tourne(&mut opened.session), vec![2.0, 4.0, 6.0]);
}

#[test]
fn le_vrai_graphe_asr_demande_sur_cuda_s_ouvre_quand_meme() {
    // Ignoré sans le graphe : il pèse des centaines de mégaoctets.
    let Some(graph) = std::env::var_os("VH_NEMOTRON_ASR_GRAPH").map(PathBuf::from) else {
        eprintln!("VH_NEMOTRON_ASR_GRAPH non défini : test sauté");
        return;
    };
    let mut asr =
        vocal_helper_rs::asr_session::NemotronAsr::open_with(&graph, Some(&[Provider::Cuda]))
            .expect("graphe ouvert");
    assert_eq!(asr.providers().last(), Some(&Provider::Cpu));
    if sans_gpu_attendu() {
        assert_eq!(asr.providers(), &[Provider::Cpu]);
    }
    let audio = vec![0.0f32; 16_000 * 2];
    asr.frames(&audio, 16_000).expect("inférence");
}
