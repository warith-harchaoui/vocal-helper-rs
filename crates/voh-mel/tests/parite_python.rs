//! Le front-end Rust produit-il les mêmes nombres que le Python ?
//!
//! C'est la seule question qui compte pour ce portage. Les deux chemins
//! alimentent le **même graphe ONNX** : s'ils ne lui donnent pas les mêmes
//! trames, le chemin mobile fabrique silencieusement d'autres probabilités, et
//! aucun test unitaire de chaque côté ne le verrait.
//!
//! La référence est produite par `vocal_helper.nemotron.log_mel`, lui-même
//! vérifié contre l'extracteur officiel du modèle à 0,006 d'écart moyen au
//! régime établi. Cette parité-ci est donc le second maillon d'une chaîne :
//! extracteur officiel → Python → Rust.
//!
//! Les cas couvrent ce qui peut différer : un spectre riche qui met tout le banc
//! à contribution, un ton pur sur la charnière de l'échelle Slaney, deux tons
//! éloignés pour détecter une fuite entre filtres voisins, et du silence, où le
//! plancher du logarithme doit être atteint exactement.

use std::collections::HashMap;

use voh_mel::{log_mel, MelSettings};

#[derive(serde::Deserialize)]
struct Case {
    audio: Vec<f32>,
    frames: Vec<Vec<f32>>,
}

/// Tolérance sur une valeur log-mel.
///
/// La référence est arrondie à cinq décimales à l'écriture, et les deux chemins
/// n'emploient pas la même transformée — pocketfft d'un côté, realfft de
/// l'autre. `1e-3` est donc large pour du bruit d'arrondi et reste cent fois
/// plus serré que l'écart de 0,006 qui sépare déjà le Python de l'extracteur
/// officiel : un vrai désaccord de convention se verrait sans ambiguïté.
const TOLERANCE: f32 = 1e-3;

fn reference() -> HashMap<String, Case> {
    let raw = include_str!("fixtures/python_reference.json");
    serde_json::from_str(raw).expect("la référence doit être lisible")
}

#[test]
fn les_deux_chemins_donnent_les_memes_trames() {
    let settings = MelSettings::default();
    for (name, case) in reference() {
        let mine = log_mel(&case.audio, settings.sample_rate, &settings)
            .unwrap_or_else(|e| panic!("{name} : {e}"));
        assert_eq!(
            mine.len(),
            case.frames.len(),
            "{name} : {} trames contre {} attendues",
            mine.len(),
            case.frames.len()
        );
        let mut worst = 0.0_f32;
        let mut worst_at = (0usize, 0usize);
        for (f, (row, expected)) in mine.iter().zip(case.frames.iter()).enumerate() {
            assert_eq!(row.len(), expected.len(), "{name} : largeur de trame");
            for (b, (&got, &want)) in row.iter().zip(expected.iter()).enumerate() {
                let gap = (got - want).abs();
                if gap > worst {
                    worst = gap;
                    worst_at = (f, b);
                }
            }
        }
        assert!(
            worst <= TOLERANCE,
            "{name} : écart maximal {worst} à la trame {} bande {} — au-delà de {TOLERANCE}",
            worst_at.0,
            worst_at.1
        );
    }
}

#[test]
fn le_silence_atteint_le_meme_plancher_des_deux_cotes() {
    let settings = MelSettings::default();
    let case = reference()
        .remove("silence")
        .expect("le cas du silence doit être dans la référence");
    let mine = log_mel(&case.audio, settings.sample_rate, &settings).unwrap();
    let floor = settings.log_guard.ln() as f32;
    for row in &case.frames {
        for &value in row {
            assert!(
                (value - floor).abs() < 1e-3,
                "la référence Python doit elle aussi être au plancher : {value}"
            );
        }
    }
    for row in &mine {
        for &value in row {
            assert!((value - floor).abs() < 1e-3, "et le Rust aussi : {value}");
        }
    }
}

#[test]
fn le_nombre_de_trames_suit_la_meme_regle_de_bord() {
    // Le remplissage et le découpage sont les deux endroits où deux
    // implémentations divergent le plus facilement d'une trame, et une trame de
    // décalage décale toute la diarisation de 10 ms.
    let settings = MelSettings::default();
    for (name, case) in reference() {
        let mine = log_mel(&case.audio, settings.sample_rate, &settings).unwrap();
        assert_eq!(
            mine.len(),
            case.frames.len(),
            "{name} : un décalage d'une trame décale toute la diarisation"
        );
    }
}
