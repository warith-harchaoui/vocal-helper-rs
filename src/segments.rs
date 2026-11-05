//! Des probabilités par trame aux tours de parole.

/// Un tour de parole : un locuteur avec le moment où il parle.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Début, en secondes.
    pub start: f64,
    /// Fin, en secondes.
    pub end: f64,
    /// Le locuteur, numéroté dans l'ordre de sa première arrivée.
    pub speaker: usize,
}

impl Segment {
    /// La durée du tour.
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// Seuiller les probabilités par trame et recoller les miettes.
///
/// Deux locuteurs simultanés donnent deux tours qui se recouvrent, ce qui est
/// voulu : le modèle sait le dire ; l'écraser perdrait l'information.
///
/// Les tours d'un même locuteur séparés de moins de `bridge_s` sont réunis,
/// parce qu'un silence de respiration n'est pas un changement de tour et qu'une
/// transcription n'a rien à faire de fragments de trois dixièmes.
///
/// `frame_s` vaut 0,01 pour la sortie haute résolution du graphe.
pub fn segments_from_probabilities(
    probabilities: &[f32],
    num_speakers: usize,
    frame_s: f64,
    threshold: f32,
    bridge_s: f64,
) -> Vec<Segment> {
    if num_speakers == 0 || probabilities.is_empty() {
        return Vec::new();
    }
    let frames = probabilities.len() / num_speakers;
    let mut out: Vec<Segment> = Vec::new();

    for speaker in 0..num_speakers {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut start: Option<usize> = None;
        for frame in 0..=frames {
            let active =
                frame < frames && probabilities[frame * num_speakers + speaker] > threshold;
            match (active, start) {
                (true, None) => start = Some(frame),
                (false, Some(from)) => {
                    spans.push((from, frame));
                    start = None;
                }
                _ => {}
            }
        }
        for (from, to) in spans {
            let (begin, finish) = (from as f64 * frame_s, to as f64 * frame_s);
            match out.last_mut() {
                Some(last) if last.speaker == speaker && begin - last.end <= bridge_s => {
                    last.end = finish;
                }
                _ => out.push(Segment {
                    start: begin,
                    end: finish,
                    speaker,
                }),
            }
        }
    }
    out.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.speaker.cmp(&b.speaker))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn le_seuillage_fait_des_tours() {
        let probabilities = [0.9_f32, 0.9, 0.1];
        let out = segments_from_probabilities(&probabilities, 1, 0.01, 0.5, 0.0);
        assert_eq!(out.len(), 1);
        assert!((out[0].start).abs() < 1e-9);
        assert!((out[0].end - 0.02).abs() < 1e-9);
    }

    #[test]
    fn un_silence_court_dans_un_tour_est_recolle() {
        let probabilities = [0.9_f32, 0.1, 0.9];
        let out = segments_from_probabilities(&probabilities, 1, 0.01, 0.5, 0.5);
        assert_eq!(
            out.len(),
            1,
            "une respiration n'est pas un changement de tour"
        );
        assert!((out[0].end - 0.03).abs() < 1e-9);
    }

    #[test]
    fn un_silence_long_n_est_pas_recolle() {
        let mut probabilities = vec![0.0_f32; 120];
        probabilities[..5].fill(0.9);
        probabilities[115..].fill(0.9);
        let out = segments_from_probabilities(&probabilities, 1, 0.01, 0.5, 0.5);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn deux_locuteurs_a_la_fois_donnent_deux_tours_qui_se_recouvrent() {
        // Le modèle sait dire le recouvrement ; l'écraser perdrait
        // l'information la plus difficile à retrouver ensuite.
        let probabilities = vec![0.9_f32; 8];
        let out = segments_from_probabilities(&probabilities, 2, 0.01, 0.5, 0.5);
        assert_eq!(out.len(), 2);
        let speakers: std::collections::HashSet<usize> = out.iter().map(|s| s.speaker).collect();
        assert_eq!(speakers.len(), 2);
        assert!(out[0].start < out[1].end && out[1].start < out[0].end);
    }

    #[test]
    fn le_silence_partout_ne_donne_rien() {
        let out = segments_from_probabilities(&vec![0.0_f32; 80], 8, 0.01, 0.5, 0.5);
        assert!(out.is_empty());
    }

    #[test]
    fn aucune_probabilite_ne_donne_rien() {
        assert!(segments_from_probabilities(&[], 8, 0.01, 0.5, 0.5).is_empty());
    }

    #[test]
    fn les_tours_sortent_tries_par_debut() {
        let mut probabilities = vec![0.0_f32; 20];
        // Le locuteur 1 parle d'abord, le locuteur 0 ensuite : la sortie doit
        // suivre le temps et non le numéro de locuteur.
        probabilities[1] = 0.9;
        probabilities[3] = 0.9;
        probabilities[12] = 0.9;
        probabilities[14] = 0.9;
        let out = segments_from_probabilities(&probabilities, 2, 0.01, 0.5, 0.5);
        assert!(out.windows(2).all(|w| w[0].start <= w[1].start));
        assert_eq!(out[0].speaker, 1, "le premier à parler doit sortir premier");
    }

    #[test]
    fn un_tour_connait_sa_duree() {
        let segment = Segment {
            start: 1.0,
            end: 2.5,
            speaker: 0,
        };
        assert!((segment.duration() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn le_seuil_decide_de_ce_qui_compte_comme_parole() {
        let probabilities = [0.4_f32, 0.4];
        assert!(segments_from_probabilities(&probabilities, 1, 0.01, 0.5, 0.0).is_empty());
        assert_eq!(
            segments_from_probabilities(&probabilities, 1, 0.01, 0.3, 0.0).len(),
            1
        );
    }
}
