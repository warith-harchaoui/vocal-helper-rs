//! Le calendrier de flux, et les deux plafonds qui le bornent.
//!
//! Les longueurs vivent dans les métadonnées du graphe plutôt que dans ce
//! fichier : un moteur mis à jour apporte les siennes, et un écart entre le code
//! et le fichier devient impossible. Les valeurs par défaut sont celles du point
//! de contrôle actuel, pour que la caisse se teste sans graphe.

use serde::{Deserialize, Serialize};

/// Le préfixe des métadonnées que l'export inscrit dans le graphe.
pub const META_PREFIX: &str = "vernacula.diar.";

/// Longueur de séquence que le graphe accepte, en trames de 80 ms, pour
/// `[cache | file | morceau]` réunis.
///
/// Déduite de l'échec et non devinée. Le message était : encodage positionnel de
/// 5000 positions pour 5144 demandées. Les positions sont à 10 ms et les trames
/// à 80 ms, donc la capacité est 5000 / 8 = 625 trames.
pub const MAX_SEQUENCE_FRAMES: usize = 625;

/// Les trois calendriers que la carte du modèle recommande, en trames de 80 ms,
/// sous la forme `(morceau, contexte droit)`.
///
/// Les latences correspondantes sont 1,04 s, 0,64 s et 0,32 s.
pub const STREAMING_MODES: [(&str, usize, usize); 3] = [
    ("low_latency", 9, 4),
    ("very_low_latency", 6, 2),
    ("ultra_low_latency", 3, 1),
];

/// Budget de latence par défaut, en secondes : le calendrier publié à basse latence.
///
/// Le plus long contexte viable, 6,4 s, est **moins cher** à qualité égale —
/// facteur temps réel de 0,166 contre 0,258, pour 0,092 de taux d'erreur contre
/// 0,090. Mais cette égalité n'existe qu'en salle anéchoïque, et une salle de
/// réunion ne l'est pas : réverbérée à 0,6 s de temps de décroissance, les deux
/// se séparent nettement — 0,350 contre 0,252 d'erreur, et un accord longue
/// portée de 0,372 contre 0,668, presque doublé. Le contexte long dilue la
/// décision quand la salle smeare le signal.
pub const DEFAULT_LATENCY_S: f64 = 1.04;

/// Le calendrier de flux, en trames de 80 ms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Schedule {
    /// Nombre maximal de locuteurs que le modèle peut distinguer.
    pub num_speakers: usize,
    /// Dimension des empreintes du cache et de la file.
    pub emb_dim: usize,
    /// Bandes attendues en entrée.
    pub n_mels: usize,
    /// Rapport entre les trames de 10 ms et celles de 80 ms.
    pub subsampling: usize,
    /// Rapport inverse, pour les probabilités à 10 ms.
    pub upsample_factor: usize,
    /// Longueur du cache de locuteurs.
    pub spkcache_len: usize,
    /// Longueur de la file.
    pub fifo_len: usize,
    /// Longueur du morceau.
    pub chunk_len: usize,
    /// Contexte droit du morceau.
    pub chunk_right_context: usize,
    /// Seuil sous lequel une trame compte comme du silence.
    pub sil_threshold: f64,
    /// Plancher appliqué aux probabilités avant le logarithme des scores.
    pub pred_score_threshold: f64,
    /// Taux du renforcement fort, qui garantit sa place à chaque locuteur.
    pub strong_boost_rate: f64,
    /// Taux du renforcement faible, qui empêche un bavard de tout occuper.
    pub weak_boost_rate: f64,
    /// Part de trames franches requise pour écarter le recouvrement.
    pub min_pos_scores_rate: f64,
    /// Prime donnée aux trames les plus récentes.
    pub scores_boost_latest: f64,
    /// Créneaux de silence réservés par locuteur dans le cache.
    pub spkcache_sil_frames_per_spk: usize,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            num_speakers: 8,
            emb_dim: 512,
            n_mels: 128,
            subsampling: 8,
            upsample_factor: 8,
            spkcache_len: 264,
            fifo_len: 40,
            chunk_len: 340,
            chunk_right_context: 40,
            sil_threshold: 0.2,
            pred_score_threshold: 0.25,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
            scores_boost_latest: 0.05,
            spkcache_sil_frames_per_spk: 1,
        }
    }
}

impl Schedule {
    /// Audio à attendre avant de traiter un morceau, en secondes.
    pub fn latency_s(&self) -> f64 {
        (self.chunk_len + self.chunk_right_context) as f64 * 0.08
    }

    /// Le graphe peut-il traiter ce calendrier à cache plein ?
    ///
    /// Deux conditions, et c'est la seconde qui mord. La séquence doit tenir dans
    /// [`MAX_SEQUENCE_FRAMES`] ; et le morceau ne doit pas dépasser la file,
    /// faute de quoi celle-ci grandit sans borne — la mise à jour lui ajoute
    /// `chunk_len` trames et n'en retire que `fifo_len`. À 281 trames de morceau
    /// elle atteint 241 au lieu de 40 et la séquence monte à 826.
    ///
    /// C'est le portage fidèle de la référence, dont la règle n'est cohérente que
    /// si le morceau n'excède pas la file : ses trois calendriers de flux
    /// publiés le respectent (9, 6, 3 trames) et son calendrier hors ligne, à
    /// 340, le viole — ce qui est exactement pourquoi ce dernier ne fonctionne
    /// pas.
    pub fn fits(&self) -> bool {
        let total = self.spkcache_len + self.fifo_len + self.chunk_len + self.chunk_right_context;
        total <= MAX_SEQUENCE_FRAMES && self.chunk_len <= self.fifo_len
    }

    /// Le même calendrier, avec un morceau et un contexte droit donnés.
    pub fn with_chunk(self, chunk_len: usize, chunk_right_context: usize) -> Self {
        Self {
            chunk_len,
            chunk_right_context,
            ..self
        }
    }

    /// Le calendrier nommé que la provenance publie, s'il existe.
    pub fn named(self, mode: &str) -> Option<Self> {
        STREAMING_MODES
            .iter()
            .find(|(name, _, _)| *name == mode)
            .map(|&(_, chunk, right)| self.with_chunk(chunk, right))
    }

    /// Le morceau le plus long que la latence demandée et le graphe permettent.
    ///
    /// Plus de contexte coûte moins cher — moins d'appels au graphe — donc la
    /// seule raison de raccourcir est la latence voulue. `None` ne demande rien
    /// et prend donc tout ce qui tient.
    pub fn within(self, latency_s: Option<f64>) -> Self {
        let right = self.chunk_right_context;
        // La file borne le morceau, et c'est la contrainte qui compte vraiment.
        let room = (MAX_SEQUENCE_FRAMES.saturating_sub(self.spkcache_len + self.fifo_len))
            .min(self.fifo_len + right);
        match latency_s {
            None => self.with_chunk(room.saturating_sub(right).max(1), right),
            Some(budget) => {
                let frames = (budget / 0.08) as usize;
                // Deux familles de candidats, et on prend le plus long des deux.
                //
                // Les calendriers publiés ont un contexte droit plus court — 4, 2
                // et 1 trames — donc ils sont les seuls à tenir dans un petit
                // budget, où le contexte droit par défaut de 40 ne passe pas. Un
                // budget large, lui, doit aller jusqu'au plus long viable : les
                // préférer par principe ignorerait la demande de l'appelant.
                let published = STREAMING_MODES
                    .iter()
                    .filter(|&&(_, chunk, rc)| chunk + rc <= frames)
                    .max_by_key(|&&(_, chunk, rc)| chunk + rc)
                    .map(|&(_, chunk, rc)| (chunk, rc));
                let computed = {
                    let longest = frames.min(room).saturating_sub(right);
                    (longest >= 1).then_some((longest, right))
                };
                match (published, computed) {
                    (Some(p), Some(c)) if c.0 + c.1 > p.0 + p.1 => self.with_chunk(c.0, c.1),
                    (Some(p), _) => self.with_chunk(p.0, p.1),
                    (None, Some(c)) => self.with_chunk(c.0, c.1),
                    // Budget intenable : le plus court plutôt que rien, parce que
                    // ne pas diariser est pire que dépasser le budget.
                    (None, None) => {
                        let &(_, chunk, rc) = STREAMING_MODES
                            .iter()
                            .min_by_key(|&&(_, chunk, rc)| chunk + rc)
                            .expect("la table n'est pas vide");
                        self.with_chunk(chunk, rc)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn les_calendriers_publies_annoncent_les_bonnes_latences() {
        let expected = [
            ("low_latency", 1.04),
            ("very_low_latency", 0.64),
            ("ultra_low_latency", 0.32),
        ];
        for (name, latency) in expected {
            let schedule = Schedule::default().named(name).unwrap();
            assert!(
                (schedule.latency_s() - latency).abs() < 1e-9,
                "{name} : {} contre {latency}",
                schedule.latency_s()
            );
        }
    }

    #[test]
    fn un_calendrier_inconnu_ne_rend_rien_plutot_qu_un_defaut() {
        assert!(Schedule::default().named("instantané").is_none());
    }

    #[test]
    fn les_calendriers_publies_tiennent_tous() {
        for (name, chunk, right) in STREAMING_MODES {
            assert!(
                Schedule::default().with_chunk(chunk, right).fits(),
                "{name} devrait tenir"
            );
        }
    }

    #[test]
    fn le_calendrier_hors_ligne_du_graphe_ne_tient_pas() {
        // 340 trames de morceau pour une file de 40 : la file grandirait sans
        // borne. C'est pourquoi ce calendrier échouait après quelques minutes,
        // le temps que le cache se remplisse.
        assert!(!Schedule::default().with_chunk(340, 40).fits());
    }

    #[test]
    fn un_morceau_plus_long_que_la_file_ne_tient_jamais() {
        let schedule = Schedule::default();
        for chunk in [41, 60, 100, 281] {
            assert!(
                !schedule.with_chunk(chunk, 40).fits(),
                "un morceau de {chunk} dépasse la file de {}",
                schedule.fifo_len
            );
        }
    }

    #[test]
    fn aucun_budget_ne_choisit_un_morceau_plus_long_que_la_file() {
        // La contrainte qui a réellement fait échouer un défaut précédent.
        let schedule = Schedule::default();
        for budget in [
            None,
            Some(60.0),
            Some(25.0),
            Some(6.4),
            Some(1.04),
            Some(0.32),
        ] {
            let chosen = schedule.within(budget);
            assert!(
                chosen.chunk_len <= schedule.fifo_len,
                "budget {budget:?} choisit un morceau de {}",
                chosen.chunk_len
            );
            assert!(
                chosen.fits(),
                "budget {budget:?} choisit un calendrier qui ne tient pas"
            );
        }
    }

    #[test]
    fn sans_budget_on_prend_le_plus_long_viable() {
        let chosen = Schedule::default().within(None);
        assert_eq!(chosen.chunk_len, 40);
        assert!((chosen.latency_s() - 6.4).abs() < 1e-9);
    }

    #[test]
    fn un_budget_d_une_seconde_rend_le_calendrier_publie() {
        let chosen = Schedule::default().within(Some(DEFAULT_LATENCY_S));
        assert_eq!((chosen.chunk_len, chosen.chunk_right_context), (9, 4));
    }

    #[test]
    fn un_budget_intenable_rend_le_plus_court_plutot_que_rien() {
        // Ne pas diariser est pire que dépasser le budget, et il faut le dire
        // par le comportement plutôt que par une erreur.
        let chosen = Schedule::default().within(Some(0.01));
        assert_eq!((chosen.chunk_len, chosen.chunk_right_context), (3, 1));
    }

    #[test]
    fn un_budget_genereux_reste_borne_par_la_file() {
        let chosen = Schedule::default().within(Some(3600.0));
        assert_eq!(chosen.chunk_len, 40);
    }
}
