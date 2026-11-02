//! La comptabilité du cache donne-t-elle les mêmes nombres qu'en Python ?
//!
//! C'est le cœur du portage. Les deux chemins pilotent le **même graphe** : le
//! graphe décide des probabilités, mais c'est cette comptabilité qui décide de
//! ce qu'il voit au pas suivant — quelles trames restent dans le cache, quelles
//! probabilités accompagnent la file. Un écart ici fait divergeur les deux
//! chemins au bout de quelques pas, sans qu'aucun test unitaire de chaque côté
//! ne le remarque.
//!
//! La référence couvre les étapes qui peuvent différer indépendamment : le score
//! d'une trame, l'écartement de ce qui n'est pas de la parole propre, le
//! renforcement, la compression, une suite de six pas de flux où le cache se
//! remplit puis se comprime, la traduction en tours de parole, et le choix du
//! calendrier pour huit budgets de latence.

use std::collections::HashMap;

use serde::Deserialize;
use voh_diar::{
    boost_top_scores, compress_cache, disable_low_scores, log_pred_scores,
    segments_from_probabilities, streaming_update, Schedule, Slot, StreamingState,
};

/// Tolérance sur une valeur de score ou d'empreinte.
///
/// La référence est arrondie à l'écriture et les deux chemins n'additionnent pas
/// dans le même ordre. `1e-5` est large pour du bruit d'arrondi et serré pour un
/// désaccord de convention, qui se verrait de plusieurs ordres de grandeur.
const TOLERANCE: f64 = 1e-5;

fn reference() -> HashMap<String, serde_json::Value> {
    serde_json::from_str(include_str!("fixtures/python_reference.json"))
        .expect("la référence doit être lisible")
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= TOLERANCE,
        "{what} : {got} contre {want} attendu"
    );
}

#[test]
fn les_scores_de_trame_concordent() {
    #[derive(Deserialize)]
    struct Case {
        predictions: Vec<f32>,
        num_speakers: usize,
        floor: f64,
        scores: Vec<f64>,
    }
    let case: Case = serde_json::from_value(reference()["log_pred_scores"].clone()).unwrap();
    let mine = log_pred_scores(&case.predictions, case.num_speakers, case.floor);
    assert_eq!(mine.len(), case.scores.len());
    for (i, (&got, &want)) in mine.iter().zip(case.scores.iter()).enumerate() {
        close(got, want, &format!("score {i}"));
    }
}

#[test]
fn l_ecartement_de_ce_qui_n_est_pas_de_la_parole_propre_concorde() {
    #[derive(Deserialize)]
    struct Case {
        predictions: Vec<f32>,
        scores: Vec<f64>,
        num_speakers: usize,
        min_positive: usize,
        /// `null` là où le Python a mis moins l'infini.
        result: Vec<Option<f64>>,
    }
    for key in ["disable_0", "disable_2", "disable_99"] {
        let case: Case = serde_json::from_value(reference()[key].clone()).unwrap();
        let mut mine = case.scores.clone();
        disable_low_scores(
            &case.predictions,
            &mut mine,
            case.num_speakers,
            case.min_positive,
        );
        for (i, (&got, want)) in mine.iter().zip(case.result.iter()).enumerate() {
            match want {
                None => assert!(
                    got.is_infinite() && got < 0.0,
                    "{key} score {i} : {got} devait être écarté"
                ),
                Some(value) => close(got, *value, &format!("{key} score {i}")),
            }
        }
    }
}

#[test]
fn le_renforcement_concorde() {
    #[derive(Deserialize)]
    struct Case {
        scores: Vec<f64>,
        num_speakers: usize,
        count: usize,
        scale: f64,
        result: Vec<Option<f64>>,
    }
    for key in ["boost_1_2", "boost_3_1", "boost_0_1"] {
        let case: Case = serde_json::from_value(reference()[key].clone()).unwrap();
        let mut mine = case.scores.clone();
        boost_top_scores(&mut mine, case.num_speakers, case.count, case.scale);
        for (i, (&got, want)) in mine.iter().zip(case.result.iter()).enumerate() {
            match want {
                None => assert!(got.is_infinite() && got < 0.0, "{key} score {i}"),
                Some(value) => close(got, *value, &format!("{key} score {i}")),
            }
        }
    }
}

#[test]
fn la_compression_du_cache_garde_les_memes_trames() {
    #[derive(Deserialize)]
    struct Case {
        embeddings: Vec<f32>,
        predictions: Vec<f32>,
        frames: usize,
        emb_dim: usize,
        num_speakers: usize,
        spkcache_len: usize,
        sil_frames: usize,
        silence: Vec<f32>,
        cache: Vec<f32>,
        kept: Vec<f32>,
    }
    let case: Case = serde_json::from_value(reference()["compress"].clone()).unwrap();
    let schedule = Schedule {
        num_speakers: case.num_speakers,
        emb_dim: case.emb_dim,
        spkcache_len: case.spkcache_len,
        spkcache_sil_frames_per_spk: case.sil_frames,
        ..Schedule::default()
    };
    let slots: Vec<Slot> = (0..case.frames)
        .map(|f| Slot {
            embedding: case.embeddings[f * case.emb_dim..(f + 1) * case.emb_dim].to_vec(),
            predictions: case.predictions[f * case.num_speakers..(f + 1) * case.num_speakers]
                .to_vec(),
        })
        .collect();
    let mine = compress_cache(&slots, &schedule, &case.silence);
    assert_eq!(mine.len(), case.spkcache_len, "longueur du cache");

    let flat: Vec<f32> = mine.iter().flat_map(|s| s.embedding.clone()).collect();
    for (i, (&got, &want)) in flat.iter().zip(case.cache.iter()).enumerate() {
        close(f64::from(got), f64::from(want), &format!("empreinte {i}"));
    }
    let preds: Vec<f32> = mine.iter().flat_map(|s| s.predictions.clone()).collect();
    for (i, (&got, &want)) in preds.iter().zip(case.kept.iter()).enumerate() {
        close(f64::from(got), f64::from(want), &format!("probabilité {i}"));
    }
}

#[test]
fn six_pas_de_flux_laissent_le_meme_etat() {
    // Le test qui compte le plus : une divergence d'un pas se propage, donc
    // suivre l'état sur plusieurs pas est la seule façon de l'attraper.
    #[derive(Deserialize)]
    struct Step {
        chunk: Vec<f32>,
        predictions: Vec<f32>,
        chunk_preds: Vec<f32>,
        cache_len: usize,
        fifo_len: usize,
        cache: Vec<f32>,
        fifo: Vec<f32>,
    }
    #[derive(Deserialize)]
    struct Case {
        num_speakers: usize,
        emb_dim: usize,
        spkcache_len: usize,
        fifo_len: usize,
        sil_frames: usize,
        steps: Vec<Step>,
    }
    let case: Case = serde_json::from_value(reference()["streaming"].clone()).unwrap();
    let schedule = Schedule {
        num_speakers: case.num_speakers,
        emb_dim: case.emb_dim,
        spkcache_len: case.spkcache_len,
        fifo_len: case.fifo_len,
        spkcache_sil_frames_per_spk: case.sil_frames,
        ..Schedule::default()
    };
    let silence = vec![0.0_f32; case.emb_dim];
    let mut state = StreamingState::new();

    for (index, step) in case.steps.iter().enumerate() {
        let chunk: Vec<Vec<f32>> = step
            .chunk
            .chunks(case.emb_dim)
            .map(<[f32]>::to_vec)
            .collect();
        let mine = streaming_update(&mut state, &chunk, &step.predictions, &schedule, &silence);
        for (i, (&got, &want)) in mine.iter().zip(step.chunk_preds.iter()).enumerate() {
            close(
                f64::from(got),
                f64::from(want),
                &format!("pas {index} probabilité {i}"),
            );
        }
        assert_eq!(state.spkcache.len(), step.cache_len, "pas {index} : cache");
        assert_eq!(state.fifo.len(), step.fifo_len, "pas {index} : file");

        let cache: Vec<f32> = state
            .spkcache
            .iter()
            .flat_map(|s| s.embedding.clone())
            .collect();
        for (i, (&got, &want)) in cache.iter().zip(step.cache.iter()).enumerate() {
            close(
                f64::from(got),
                f64::from(want),
                &format!("pas {index} cache {i}"),
            );
        }
        let fifo: Vec<f32> = state
            .fifo
            .iter()
            .flat_map(|s| s.embedding.clone())
            .collect();
        for (i, (&got, &want)) in fifo.iter().zip(step.fifo.iter()).enumerate() {
            close(
                f64::from(got),
                f64::from(want),
                &format!("pas {index} file {i}"),
            );
        }
    }
}

#[test]
fn les_tours_de_parole_concordent() {
    #[derive(Deserialize)]
    struct Case {
        probabilities: Vec<f32>,
        num_speakers: usize,
        threshold: f32,
        bridge_s: f64,
        /// `(début, fin, locuteur)`.
        segments: Vec<(f64, f64, usize)>,
    }
    for key in ["segments_0.05", "segments_0.5"] {
        let case: Case = serde_json::from_value(reference()[key].clone()).unwrap();
        let mine = segments_from_probabilities(
            &case.probabilities,
            case.num_speakers,
            0.01,
            case.threshold,
            case.bridge_s,
        );
        assert_eq!(mine.len(), case.segments.len(), "{key} : nombre de tours");
        for (i, (got, want)) in mine.iter().zip(case.segments.iter()).enumerate() {
            close(got.start, want.0, &format!("{key} tour {i} début"));
            close(got.end, want.1, &format!("{key} tour {i} fin"));
            assert_eq!(got.speaker, want.2, "{key} tour {i} locuteur");
        }
    }
}

#[test]
fn le_choix_du_calendrier_concorde_pour_tous_les_budgets() {
    #[derive(Deserialize)]
    struct Row {
        latency_s: Option<f64>,
        chunk: usize,
        right: usize,
    }
    let rows: Vec<Row> = serde_json::from_value(reference()["schedules"].clone()).unwrap();
    assert!(
        rows.len() >= 8,
        "la référence doit couvrir plusieurs budgets"
    );
    for row in rows {
        let mine = Schedule::default().within(row.latency_s);
        assert_eq!(
            (mine.chunk_len, mine.chunk_right_context),
            (row.chunk, row.right),
            "budget {:?}",
            row.latency_s
        );
    }
}
