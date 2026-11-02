//! La tenue du cache de locuteurs et de la file, autour du graphe.
//!
//! Le graphe ONNX est **un pas de flux**, pas un diariseur de fichier : il
//! reçoit `[cache | file | morceau]` et rend des probabilités. Tout ce qui
//! décide de la qualité du suivi vit autour de lui — quelles trames entrent dans
//! la file, lesquelles en sortent, comment le cache se comprime quand il déborde
//! — et c'est le portage de `SortformerModules.streaming_update` de la
//! référence.
//!
//! Tout ici est de l'arithmétique sur des tableaux : ni modèle, ni réseau, ni
//! ONNX. C'est donc entièrement vérifiable, et c'est vérifié — y compris contre
//! le chemin Python, dans `tests/parite_python.rs`.

use crate::schedule::Schedule;

/// Une observation gardée dans le cache, et sa probabilité par locuteur.
#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
    /// L'empreinte, de dimension `emb_dim`.
    pub embedding: Vec<f32>,
    /// Les probabilités de chaque locuteur sur cette trame.
    pub predictions: Vec<f32>,
}

/// Le cache de locuteurs et la file, entre deux pas de flux.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamingState {
    /// Ce que le flux a retenu des locuteurs.
    pub spkcache: Vec<Slot>,
    /// Le contexte récent, en attente de rejoindre le cache.
    pub fifo: Vec<Slot>,
    /// Le cache a-t-il déjà été rempli une fois ?
    ///
    /// Au premier remplissage, les probabilités du cache viennent de ce que le
    /// graphe vient d'en dire plutôt que d'un historique qui n'existe pas.
    pub cache_seen: bool,
}

impl StreamingState {
    /// Un état de départ : tout vide, rien de retenu.
    pub fn new() -> Self {
        Self::default()
    }

    /// Les empreintes du cache, à plat, pour l'entrée du graphe.
    pub fn spkcache_flat(&self, emb_dim: usize) -> Vec<f32> {
        flatten(&self.spkcache, emb_dim)
    }

    /// Les empreintes de la file, à plat.
    pub fn fifo_flat(&self, emb_dim: usize) -> Vec<f32> {
        flatten(&self.fifo, emb_dim)
    }
}

fn flatten(slots: &[Slot], emb_dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(slots.len() * emb_dim);
    for slot in slots {
        out.extend_from_slice(&slot.embedding);
    }
    out
}

/// Le score d'une trame pour chaque locuteur, à partir des probabilités.
///
/// Conçu pour être élevé quand **un seul** locuteur parle avec confiance : il
/// récompense la probabilité du locuteur considéré et pénalise celle des autres.
/// C'est ce qui fait qu'un cache rempli par ces scores contient de la parole
/// propre et non du recouvrement.
pub fn log_pred_scores(predictions: &[f32], num_speakers: usize, floor: f64) -> Vec<f64> {
    let frames = predictions.len() / num_speakers;
    let mut out = vec![0.0; frames * num_speakers];
    for f in 0..frames {
        let row = &predictions[f * num_speakers..(f + 1) * num_speakers];
        let log_1p: f64 = row
            .iter()
            .map(|&p| (1.0 - f64::from(p)).max(floor).ln())
            .sum();
        for (s, &p) in row.iter().enumerate() {
            let log_p = f64::from(p).max(floor).ln();
            let log_1 = (1.0 - f64::from(p)).max(floor).ln();
            out[f * num_speakers + s] = log_p - log_1 + log_1p - 0.5_f64.ln();
        }
    }
    out
}

/// Écarter du cache ce qui n'est pas de la parole propre.
///
/// Deux coups de balai. Ce qui n'est pas de la parole pour ce locuteur est
/// écarté définitivement. Et ce qui a un score non positif — en général du
/// recouvrement — l'est aussi, **à condition** que le locuteur dispose déjà
/// d'assez de trames franches : sans cette condition, un locuteur qui ne parle
/// jamais seul disparaîtrait du cache.
pub fn disable_low_scores(
    predictions: &[f32],
    scores: &mut [f64],
    num_speakers: usize,
    min_positive: usize,
) {
    let frames = predictions.len() / num_speakers;
    // L'ORDRE COMPTE, et il est le même qu'en Python : on masque d'abord ce qui
    // n'est pas de la parole, PUIS on compte les trames franches sur le tableau
    // déjà masqué. Compter avant donnerait un décompte plus élevé, donc
    // écarterait du recouvrement là où le locuteur n'a en fait pas de quoi.
    for f in 0..frames {
        for s in 0..num_speakers {
            let i = f * num_speakers + s;
            if predictions[i] <= 0.5 {
                scores[i] = f64::NEG_INFINITY;
            }
        }
    }
    for s in 0..num_speakers {
        let positives = (0..frames)
            .filter(|&f| scores[f * num_speakers + s] > 0.0)
            .count();
        if positives < min_positive {
            continue;
        }
        for f in 0..frames {
            let i = f * num_speakers + s;
            // Le recouvrement — score non positif sur de la parole — n'est
            // écarté que si le locuteur a de quoi par ailleurs.
            if predictions[i] > 0.5 && scores[i] <= 0.0 {
                scores[i] = f64::NEG_INFINITY;
            }
        }
    }
}

/// Relever les `count` meilleurs scores de chaque locuteur.
///
/// Appliqué deux fois par la compression, avec deux intensités : une forte, pour
/// garantir à chaque locuteur un minimum de place dans le cache, et une faible,
/// pour empêcher un bavard d'occuper tout l'espace. Les trames déjà écartées
/// restent à moins l'infini.
pub fn boost_top_scores(scores: &mut [f64], num_speakers: usize, count: usize, scale: f64) {
    if count == 0 {
        return;
    }
    let frames = scores.len() / num_speakers;
    let bonus = -scale * 0.5_f64.ln();
    for s in 0..num_speakers {
        let mut order: Vec<usize> = (0..frames).collect();
        order.sort_by(|&a, &b| {
            scores[b * num_speakers + s]
                .partial_cmp(&scores[a * num_speakers + s])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &f in order.iter().take(count.min(frames)) {
            let i = f * num_speakers + s;
            if scores[i].is_finite() {
                scores[i] += bonus;
            }
        }
    }
}

/// Ramener un cache débordant à sa taille, en gardant ce qui compte.
///
/// On garde les trames dont les scores disent qu'elles portent de la parole
/// propre, en réservant à chaque locuteur sa part et en laissant quelques
/// créneaux de silence par locuteur. Les créneaux vides reçoivent l'empreinte de
/// silence **apprise** que l'export transporte, et non la moyenne des trames
/// silencieuses — c'est l'un des deux écarts de ce point de contrôle par rapport
/// à Streaming Sortformer 2.x.
pub fn compress_cache(slots: &[Slot], schedule: &Schedule, silence: &[f32]) -> Vec<Slot> {
    let n = schedule.num_speakers;
    let frames = slots.len();
    if frames <= schedule.spkcache_len {
        return slots.to_vec();
    }
    let per_speaker = (schedule.spkcache_len / n)
        .saturating_sub(schedule.spkcache_sil_frames_per_spk)
        .max(1);

    let predictions: Vec<f32> = slots.iter().flat_map(|s| s.predictions.clone()).collect();
    let mut scores = log_pred_scores(&predictions, n, schedule.pred_score_threshold);
    disable_low_scores(
        &predictions,
        &mut scores,
        n,
        (per_speaker as f64 * schedule.min_pos_scores_rate).floor() as usize,
    );
    if schedule.scores_boost_latest > 0.0 && frames > schedule.spkcache_len {
        for f in schedule.spkcache_len..frames {
            for s in 0..n {
                scores[f * n + s] += schedule.scores_boost_latest;
            }
        }
    }
    boost_top_scores(
        &mut scores,
        n,
        (per_speaker as f64 * schedule.strong_boost_rate).floor() as usize,
        2.0,
    );
    boost_top_scores(
        &mut scores,
        n,
        (per_speaker as f64 * schedule.weak_boost_rate).floor() as usize,
        1.0,
    );

    // Les scores sont mis à plat PAR LOCUTEUR avant le tri, si bien qu'une même
    // trame peut être retenue au titre de deux locuteurs ; le modulo rend
    // ensuite l'indice de trame. L'ordre chronologique est rétabli, parce que le
    // cache est relu comme une séquence et non comme un sac.
    let padded_frames = frames + schedule.spkcache_sil_frames_per_spk;
    let mut flat: Vec<(f64, usize)> = Vec::with_capacity(padded_frames * n);
    for s in 0..n {
        for f in 0..frames {
            flat.push((scores[f * n + s], s * padded_frames + f));
        }
        for extra in 0..schedule.spkcache_sil_frames_per_spk {
            flat.push((f64::INFINITY, s * padded_frames + frames + extra));
        }
    }
    flat.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    // L'ordre retenu est celui des indices À PLAT, pas celui des trames : les
    // scores ont été mis à plat par locuteur, donc trier les indices range
    // d'abord les trames du locuteur 0, puis son créneau de silence, puis celles
    // du locuteur 1, et ainsi de suite. Les créneaux de silence s'intercalent
    // ainsi au lieu de finir en queue — c'est la convention de la référence, et
    // le cache est relu dans cet ordre.
    let sentinel = usize::MAX;
    let mut chosen: Vec<usize> = flat
        .iter()
        .take(schedule.spkcache_len)
        .map(|&(score, index)| {
            if score.is_infinite() && score < 0.0 {
                sentinel
            } else {
                index
            }
        })
        .collect();
    chosen.sort_unstable();
    let chosen: Vec<(usize, bool)> = chosen
        .into_iter()
        .map(|index| {
            if index == sentinel {
                return (0, true);
            }
            let frame = index % padded_frames;
            if frame >= frames {
                (0, true)
            } else {
                (frame, false)
            }
        })
        .collect();

    chosen
        .into_iter()
        .map(|(frame, disabled)| {
            if disabled {
                Slot {
                    embedding: silence.to_vec(),
                    predictions: vec![0.0; n],
                }
            } else {
                slots[frame].clone()
            }
        })
        .collect()
}

/// Faire entrer un morceau dans la file, et la file dans le cache.
///
/// Le portage de `streaming_update`, pour un seul flux. Les probabilités reçues
/// couvrent `[cache | file | morceau]` : on en découpe la part de la file et
/// celle du morceau, le morceau rejoint la file, et ce que la file pousse dehors
/// rejoint le cache — qu'on comprime s'il déborde.
///
/// Rend les probabilités du seul morceau.
pub fn streaming_update(
    state: &mut StreamingState,
    chunk_embeddings: &[Vec<f32>],
    predictions: &[f32],
    schedule: &Schedule,
    silence: &[f32],
) -> Vec<f32> {
    let n = schedule.num_speakers;
    let cache_len = state.spkcache.len();
    let fifo_len = state.fifo.len();
    let chunk_len = chunk_embeddings.len();

    let row = |frame: usize| -> Vec<f32> { predictions[frame * n..(frame + 1) * n].to_vec() };

    // Les probabilités que le graphe vient de donner à la file, qui remplacent
    // celles qu'elle portait : le modèle a revu ces trames avec plus de contexte.
    for (i, slot) in state.fifo.iter_mut().enumerate() {
        slot.predictions = row(cache_len + i);
    }
    if !state.cache_seen {
        for (i, slot) in state.spkcache.iter_mut().enumerate() {
            slot.predictions = row(i);
        }
    }

    let chunk_preds: Vec<f32> =
        predictions[(cache_len + fifo_len) * n..(cache_len + fifo_len + chunk_len) * n].to_vec();
    for (i, embedding) in chunk_embeddings.iter().enumerate() {
        state.fifo.push(Slot {
            embedding: embedding.clone(),
            predictions: chunk_preds[i * n..(i + 1) * n].to_vec(),
        });
    }

    if fifo_len + chunk_len > schedule.fifo_len {
        let pop = chunk_len
            .max(1)
            .min(schedule.fifo_len)
            .min(state.fifo.len());
        let popped: Vec<Slot> = state.fifo.drain(..pop).collect();
        state.spkcache.extend(popped);
        state.cache_seen = true;
        if state.spkcache.len() > schedule.spkcache_len {
            state.spkcache = compress_cache(&state.spkcache, schedule, silence);
        }
    }

    chunk_preds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(value: f32, predictions: Vec<f32>) -> Slot {
        Slot {
            embedding: vec![value; 4],
            predictions,
        }
    }

    fn small() -> Schedule {
        Schedule {
            num_speakers: 2,
            emb_dim: 4,
            spkcache_len: 4,
            fifo_len: 4,
            spkcache_sil_frames_per_spk: 0,
            ..Schedule::default()
        }
    }

    #[test]
    fn un_locuteur_seul_et_sur_marque_plus_que_deux_qui_parlent_ensemble() {
        let alone = log_pred_scores(&[0.95, 0.02], 2, 0.25);
        let together = log_pred_scores(&[0.95, 0.95], 2, 0.25);
        assert!(alone[0] > together[0]);
    }

    #[test]
    fn le_plancher_garde_les_scores_finis() {
        let scores = log_pred_scores(&[0.0, 1.0], 2, 0.25);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn ce_qui_n_est_pas_de_la_parole_est_ecarte() {
        let predictions = [0.9_f32, 0.2];
        let mut scores = vec![0.0; 2];
        disable_low_scores(&predictions, &mut scores, 2, 99);
        assert!(scores[1].is_infinite() && scores[1] < 0.0);
        assert!(scores[0].is_finite());
    }

    #[test]
    fn le_recouvrement_n_est_ecarte_que_si_le_locuteur_a_de_quoi() {
        // Sans cette condition, un locuteur qui ne parle jamais seul
        // disparaîtrait du cache.
        let predictions = [0.9_f32, 0.9, 0.9];
        let mut scores = vec![-1.0, -1.0, -1.0];
        disable_low_scores(&predictions, &mut scores, 1, 1);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn le_renforcement_ne_ressuscite_pas_une_trame_ecartee() {
        let mut scores = vec![f64::NEG_INFINITY, 1.0];
        boost_top_scores(&mut scores, 1, 2, 1.0);
        assert!(scores[0].is_infinite() && scores[0] < 0.0);
    }

    #[test]
    fn le_renforcement_leve_exactement_le_nombre_demande() {
        let mut scores = vec![0.0, 1.0, 2.0];
        boost_top_scores(&mut scores, 1, 2, 1.0);
        assert_eq!(scores[0], 0.0, "le plus faible ne doit pas bouger");
        assert!(scores[1] > 1.0);
        assert!(scores[2] > 2.0);
    }

    #[test]
    fn ne_rien_renforcer_ne_change_rien() {
        let mut scores = vec![0.0, 1.0];
        let before = scores.clone();
        boost_top_scores(&mut scores, 1, 0, 1.0);
        assert_eq!(scores, before);
    }

    #[test]
    fn la_compression_ramene_le_cache_a_sa_taille() {
        let schedule = Schedule {
            spkcache_len: 3,
            ..small()
        };
        let slots: Vec<Slot> = (0..9).map(|i| slot(i as f32, vec![0.9, 0.0])).collect();
        let out = compress_cache(&slots, &schedule, &[0.0; 4]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn un_cache_qui_ne_deborde_pas_est_rendu_tel_quel() {
        let schedule = small();
        let slots: Vec<Slot> = (0..3).map(|i| slot(i as f32, vec![0.9, 0.0])).collect();
        assert_eq!(compress_cache(&slots, &schedule, &[0.0; 4]), slots);
    }

    #[test]
    fn la_compression_donne_une_place_a_chaque_locuteur() {
        // Le rôle du renforcement fort : personne ne disparaît du cache, même
        // face à un bavard.
        let schedule = Schedule {
            spkcache_len: 2,
            ..small()
        };
        let slots = vec![
            slot(1.0, vec![0.9, 0.0]),
            slot(1.0, vec![0.9, 0.0]),
            slot(1.0, vec![0.9, 0.0]),
            slot(2.0, vec![0.0, 0.9]),
        ];
        let out = compress_cache(&slots, &schedule, &[0.0; 4]);
        let loudest: Vec<usize> = out
            .iter()
            .map(|s| {
                s.predictions
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap()
            })
            .collect();
        let mut sorted = loudest.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![0, 1],
            "les deux voix doivent survivre : {loudest:?}"
        );
    }

    #[test]
    fn la_compression_preserve_l_ordre_chronologique() {
        // Le cache est relu comme une séquence, pas comme un sac.
        let schedule = Schedule {
            num_speakers: 1,
            spkcache_len: 3,
            ..small()
        };
        let slots: Vec<Slot> = (0..6).map(|i| slot(i as f32, vec![0.9])).collect();
        let out = compress_cache(&slots, &schedule, &[0.0; 4]);
        let firsts: Vec<f32> = out.iter().map(|s| s.embedding[0]).collect();
        let mut sorted = firsts.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            firsts, sorted,
            "l'ordre doit être chronologique : {firsts:?}"
        );
    }

    #[test]
    fn les_creneaux_de_silence_recoivent_l_empreinte_apprise() {
        // L'écart assumé de ce point de contrôle : une empreinte apprise, pas
        // une moyenne des trames silencieuses.
        let schedule = Schedule {
            num_speakers: 1,
            spkcache_len: 4,
            spkcache_sil_frames_per_spk: 2,
            ..small()
        };
        let slots: Vec<Slot> = (0..5).map(|i| slot(i as f32, vec![0.9])).collect();
        let silence = vec![7.0_f32; 4];
        let out = compress_cache(&slots, &schedule, &silence);
        let filled = out.iter().filter(|s| s.embedding == silence).count();
        assert!(filled >= 1, "au moins un créneau de silence attendu");
        for slot in out.iter().filter(|s| s.embedding == silence) {
            assert!(
                slot.predictions.iter().all(|&p| p == 0.0),
                "un créneau de silence ne doit porter aucune probabilité"
            );
        }
    }

    #[test]
    fn la_file_se_remplit_avant_de_pousser_vers_le_cache() {
        let schedule = small();
        let mut state = StreamingState::new();
        let chunk = vec![vec![1.0_f32; 4], vec![1.0; 4]];
        let predictions = vec![0.9_f32; 2 * 2];
        let out = streaming_update(&mut state, &chunk, &predictions, &schedule, &[0.0; 4]);
        assert_eq!(state.fifo.len(), 2);
        assert!(state.spkcache.is_empty());
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn la_file_pousse_dans_le_cache_quand_elle_deborde() {
        let schedule = Schedule {
            fifo_len: 2,
            ..small()
        };
        let mut state = StreamingState::new();
        let silence = vec![0.0_f32; 4];
        for _ in 0..3 {
            let chunk = vec![vec![1.0_f32; 4], vec![1.0; 4]];
            let frames = state.spkcache.len() + state.fifo.len() + 2;
            let predictions = vec![0.9_f32; frames * 2];
            streaming_update(&mut state, &chunk, &predictions, &schedule, &silence);
        }
        assert!(!state.spkcache.is_empty(), "la file n'a rien poussé");
        assert!(state.fifo.len() <= schedule.fifo_len);
    }

    #[test]
    fn le_cache_ne_depasse_jamais_sa_longueur() {
        let schedule = Schedule {
            fifo_len: 2,
            spkcache_len: 4,
            ..small()
        };
        let mut state = StreamingState::new();
        let silence = vec![0.0_f32; 4];
        for step in 0..20 {
            let chunk = vec![vec![step as f32; 4], vec![step as f32 + 0.5; 4]];
            let frames = state.spkcache.len() + state.fifo.len() + 2;
            let predictions = vec![0.9_f32; frames * 2];
            streaming_update(&mut state, &chunk, &predictions, &schedule, &silence);
            assert!(
                state.spkcache.len() <= schedule.spkcache_len,
                "pas {}",
                step
            );
            assert!(state.fifo.len() <= schedule.fifo_len, "pas {}", step);
        }
    }

    #[test]
    fn les_probabilites_rendues_sont_celles_du_morceau() {
        let schedule = small();
        let mut state = StreamingState::new();
        let chunk = vec![vec![1.0_f32; 4]];
        let predictions = vec![0.1_f32, 0.2];
        let out = streaming_update(&mut state, &chunk, &predictions, &schedule, &[0.0; 4]);
        assert_eq!(out, vec![0.1, 0.2]);
    }

    #[test]
    fn l_etat_a_plat_a_la_taille_que_le_graphe_attend() {
        let schedule = small();
        let mut state = StreamingState::new();
        let chunk = vec![vec![1.0_f32; 4], vec![2.0; 4]];
        let predictions = vec![0.9_f32; 4];
        streaming_update(&mut state, &chunk, &predictions, &schedule, &[0.0; 4]);
        assert_eq!(state.fifo_flat(4).len(), 2 * 4);
        assert_eq!(state.spkcache_flat(4).len(), 0);
    }
}
