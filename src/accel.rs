//! Choisir le moteur d'exécution en le **mesurant**, pas en le supposant.
//!
//! # Pourquoi ce module existe
//!
//! Le raisonnement évident sur les accélérateurs est faux : on l'a payé. Sur
//! Apple Silicon, ONNX Runtime n'a pas de moteur Metal : l'accélération passe
//! donc par CoreML ; le mettre en tête semblait aller de soi. Mesuré sur le
//! graphe de diarisation, CoreML donne un facteur temps réel de **1,257** —
//! c'est-à-dire plus lent que le direct, donc inutilisable — contre **0,528**
//! pour le CPU seul. La cause se lit dans le journal de la session : CoreML ne
//! prend que 1193 nœuds sur 2827 et les découpe en **316 partitions**, donc
//! autant d'allers-retours entre deux moteurs à chaque pas de flux. Un
//! transformeur à axes dynamiques est précisément le cas où le déport partiel
//! coûte plus qu'il ne rapporte.
//!
//! Le second réglage « évident » était fautif aussi : forcer huit fils sur une
//! machine à huit cœurs performants mesure 0,953 contre 0,528 en laissant le
//! runtime choisir.
//!
//! La leçon n'est pas « CoreML est mauvais » — il sera excellent sur un autre
//! graphe ou dans une version qui partitionne mieux. La leçon est qu'un ordre
//! codé en dur sera faux quelque part ; il faut donc **essayer sur
//! l'appareil** puis retenir le verdict. C'est ce que ce module organise.
//!
//! Le choix lui-même est une politique pure — pas de modèle, pas de réseau, pas
//! d'ONNX — donc elle se vérifie entièrement par des tests, sans la feature
//! `onnx`, sur une machine qui n'a pas téléchargé le runtime.
//!
//! **Pas encore branché** : `NemotronDiar` et `NemotronAsr` ouvrent leur graphe
//! sur le moteur par défaut d'ONNX Runtime, le CPU, sans consulter ce module.
//! Chronométrer les moteurs sur l'appareil puis appliquer le verdict reste à
//! écrire.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Un moteur d'exécution d'ONNX Runtime.
///
/// L'énumération couvre ce que le runtime propose selon la plateforme ; une
/// variante indisponible à la compilation ou à l'exécution est simplement sautée
/// plutôt que de faire échouer le chargement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    /// Toujours là ; plus souvent gagnant qu'on ne le croit.
    Cpu,
    /// Apple : GPU et moteur neuronal. Mesuré perdant sur le graphe de
    /// diarisation, à cause du découpage en partitions.
    CoreMl,
    /// NVIDIA. Couverture d'opérateurs complète, donc pas de découpage à payer.
    Cuda,
    /// NVIDIA, compilé. Démarrage plus long, régime établi plus rapide.
    TensorRt,
    /// Windows, tous GPU.
    DirectMl,
    /// Android.
    Nnapi,
    /// Qualcomm.
    Qnn,
    /// Intel.
    OpenVino,
}

impl Provider {
    /// Le nom que le runtime emploie, pour les journaux et la configuration.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "CPUExecutionProvider",
            Self::CoreMl => "CoreMLExecutionProvider",
            Self::Cuda => "CUDAExecutionProvider",
            Self::TensorRt => "TensorrtExecutionProvider",
            Self::DirectMl => "DmlExecutionProvider",
            Self::Nnapi => "NnapiExecutionProvider",
            Self::Qnn => "QNNExecutionProvider",
            Self::OpenVino => "OpenVINOExecutionProvider",
        }
    }

    /// Les moteurs qui ont un sens sur la plateforme visée, dans l'ordre à
    /// **essayer** — pas dans l'ordre à croire.
    ///
    /// L'ordre place en tête ceux dont la couverture d'opérateurs est complète,
    /// parce qu'un déport partiel paie un aller-retour par partition. Le CPU
    /// vient tôt pour la même raison : il ne découpe rien. C'est un ordre de
    /// départ pour la mesure : [`choose`] le remplace par le verdict mesuré.
    pub fn candidates() -> Vec<Self> {
        let mut out = Vec::new();
        if cfg!(target_os = "android") {
            out.push(Self::Nnapi);
        }
        if cfg!(target_os = "windows") {
            out.push(Self::DirectMl);
        }
        if cfg!(any(target_os = "linux", target_os = "windows")) {
            out.push(Self::TensorRt);
            out.push(Self::Cuda);
            out.push(Self::OpenVino);
        }
        out.push(Self::Cpu);
        if cfg!(any(target_os = "macos", target_os = "ios")) {
            // Après le CPU ; le commentaire du module dit pourquoi : mesuré
            // 2,4 fois plus lent sur ce graphe, à cause de 316 partitions.
            out.push(Self::CoreMl);
        }
        out
    }
}

/// Ce qu'une mesure sur l'appareil a appris d'un moteur.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Timing {
    /// Le moteur mesuré.
    pub provider: Provider,
    /// Secondes de calcul par seconde d'audio. Sous 1, le direct tient.
    pub real_time_factor: f64,
}

/// Le verdict d'une campagne de mesure, tel qu'on le retient d'une session à l'autre.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// Le moteur retenu.
    pub chosen: Provider,
    /// Tout ce qui a été mesuré, pour que le choix soit relisible.
    pub timings: Vec<Timing>,
    /// L'empreinte du graphe mesuré : un autre graphe mérite une autre mesure.
    pub graph_digest: String,
}

impl Verdict {
    /// Le facteur temps réel du moteur retenu.
    pub fn real_time_factor(&self) -> Option<f64> {
        self.timings
            .iter()
            .find(|t| t.provider == self.chosen)
            .map(|t| t.real_time_factor)
    }

    /// Le verdict vaut-il encore pour ce graphe ?
    ///
    /// Un verdict retenu pour un graphe ne dit rien d'un autre : le découpage en
    /// partitions dépend des opérateurs, donc du modèle. C'est précisément
    /// l'erreur qu'un ordre codé en dur commet.
    pub fn applies_to(&self, graph_digest: &str) -> bool {
        self.graph_digest == graph_digest
    }
}

/// Choisir d'après ce qui a été mesuré, en préférant la marge au classement.
///
/// On retient le plus rapide plutôt que le premier d'un ordre supposé. Une mesure
/// absente vaut « non disponible », pas « mauvais » : un moteur qu'on n'a pas pu
/// essayer ne doit pas être écarté pour autant, il est simplement en bas.
///
/// # Exemples
///
/// Le plus rapide gagne, même si l'ordre de départ le mettait dernier.
///
/// ```
/// use vocal_helper_rs::{choose, Provider, Timing};
/// let timings = vec![
///     Timing { provider: Provider::CoreMl, real_time_factor: 1.257 },
///     Timing { provider: Provider::Cpu, real_time_factor: 0.528 },
/// ];
/// assert_eq!(choose(&timings), Some(Provider::Cpu));
/// ```
///
/// Sans aucune mesure, on ne décide pas.
///
/// ```
/// use vocal_helper_rs::choose;
/// assert_eq!(choose(&[]), None);
/// ```
pub fn choose(timings: &[Timing]) -> Option<Provider> {
    timings
        .iter()
        .filter(|t| t.real_time_factor.is_finite() && t.real_time_factor > 0.0)
        .min_by(|a, b| {
            a.real_time_factor
                .partial_cmp(&b.real_time_factor)
                .expect("les valeurs non finies sont filtrées")
        })
        .map(|t| t.provider)
}

/// Un moteur tient-il le direct, en comptant ce qui tourne à côté ?
///
/// La contrainte en direct n'est pas la latence d'un étage mais le facteur temps
/// réel **total** du système : les étages travaillent à des cadences
/// différentes, donc ce qui s'additionne est le travail sur une même unité de
/// calcul et ce qui se recouvre est le travail sur des unités distinctes.
///
/// # Exemples
///
/// ```
/// use vocal_helper_rs::holds_live;
/// // Diarisation à 0,528 plus transcription à 0,164 sur le même CPU.
/// assert!(holds_live(0.528, 0.164));
/// // La même diarisation sous CoreML, mesurée à 1,257, ne tient pas seule.
/// assert!(!holds_live(1.257, 0.0));
/// ```
pub fn holds_live(stage_rtf: f64, other_rtf_same_unit: f64) -> bool {
    stage_rtf + other_rtf_same_unit < 1.0
}

/// La durée d'un échantillon d'essai, pour une mesure qui veut dire quelque chose.
///
/// Trop court, on mesure le démarrage de la session ; trop long, on fait
/// attendre au premier lancement. Trente secondes couvrent plusieurs pas de flux
/// au calendrier par défaut, donc le régime établi — et c'est le régime établi
/// qui compte, puisque c'est lui qui tourne pendant une heure de réunion.
pub const PROBE_AUDIO: Duration = Duration::from_secs(30);

#[cfg(test)]
mod tests {
    use super::*;

    fn timing(provider: Provider, rtf: f64) -> Timing {
        Timing {
            provider,
            real_time_factor: rtf,
        }
    }

    #[test]
    fn le_plus_rapide_gagne_meme_s_il_etait_dernier_dans_l_ordre_de_depart() {
        // Le cas réel : CoreML est après le CPU dans `candidates`, mais c'est la
        // MESURE qui doit décider, pas la liste.
        let chosen = choose(&[
            timing(Provider::CoreMl, 1.257),
            timing(Provider::Cpu, 0.528),
        ]);
        assert_eq!(chosen, Some(Provider::Cpu));
    }

    #[test]
    fn un_moteur_plus_rapide_est_retenu_quel_qu_il_soit() {
        // L'inverse doit marcher aussi : rien dans le code ne doit préférer le
        // CPU par principe.
        let chosen = choose(&[timing(Provider::Cpu, 0.528), timing(Provider::Cuda, 0.031)]);
        assert_eq!(chosen, Some(Provider::Cuda));
    }

    #[test]
    fn sans_mesure_on_ne_decide_pas() {
        assert_eq!(choose(&[]), None);
    }

    #[test]
    fn une_mesure_absurde_est_ignoree_plutot_que_crue() {
        // Un essai qui a échoué peut rendre zéro ou l'infini ; les prendre au
        // mot choisirait le moteur qui n'a pas tourné.
        let chosen = choose(&[
            timing(Provider::CoreMl, 0.0),
            timing(Provider::Cuda, f64::INFINITY),
            timing(Provider::Cpu, 0.528),
        ]);
        assert_eq!(chosen, Some(Provider::Cpu));
    }

    #[test]
    fn le_cpu_est_toujours_candidat() {
        // Quelle que soit la plateforme : c'est le seul qui ne peut pas manquer :
        // c'est ce qui garantit qu'on diarise toujours.
        assert!(Provider::candidates().contains(&Provider::Cpu));
    }

    #[test]
    fn les_candidats_sont_sans_doublon() {
        let candidates = Provider::candidates();
        let mut seen = std::collections::HashSet::new();
        for provider in &candidates {
            assert!(seen.insert(*provider), "{provider:?} proposé deux fois");
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn sur_apple_le_cpu_passe_devant_coreml() {
        // L'ordre de départ encode la mesure : CoreML découpe ce graphe en 316
        // partitions et paie un aller-retour par partition.
        let candidates = Provider::candidates();
        let cpu = candidates.iter().position(|p| *p == Provider::Cpu);
        let coreml = candidates.iter().position(|p| *p == Provider::CoreMl);
        assert!(cpu < coreml, "le CPU doit être essayé avant CoreML");
    }

    #[test]
    fn chaque_moteur_a_le_nom_que_le_runtime_attend() {
        for provider in Provider::candidates() {
            let name = provider.as_str();
            assert!(
                name.ends_with("ExecutionProvider"),
                "{name} ne ressemble pas à un nom de moteur"
            );
        }
    }

    #[test]
    fn un_verdict_ne_vaut_que_pour_son_graphe() {
        // L'erreur qu'un ordre codé en dur commet : le découpage dépend des
        // opérateurs, donc du modèle.
        let verdict = Verdict {
            chosen: Provider::Cpu,
            timings: vec![timing(Provider::Cpu, 0.528)],
            graph_digest: "abc".into(),
        };
        assert!(verdict.applies_to("abc"));
        assert!(!verdict.applies_to("def"));
    }

    #[test]
    fn un_verdict_rend_le_facteur_de_son_choix() {
        let verdict = Verdict {
            chosen: Provider::Cpu,
            timings: vec![
                timing(Provider::CoreMl, 1.257),
                timing(Provider::Cpu, 0.528),
            ],
            graph_digest: "abc".into(),
        };
        assert_eq!(verdict.real_time_factor(), Some(0.528));
    }

    #[test]
    fn le_budget_du_direct_additionne_ce_qui_partage_une_unite() {
        assert!(holds_live(0.528, 0.164));
        assert!(!holds_live(0.9, 0.164));
    }

    #[test]
    fn un_moteur_plus_lent_que_le_direct_ne_tient_pas_meme_seul() {
        // Le chiffre mesuré pour CoreML sur ce graphe.
        assert!(!holds_live(1.257, 0.0));
    }

    #[test]
    fn un_verdict_fait_l_aller_retour_en_json() {
        // Il est retenu d'une session à l'autre, donc il doit survivre au disque.
        let verdict = Verdict {
            chosen: Provider::CoreMl,
            timings: vec![timing(Provider::CoreMl, 1.257)],
            graph_digest: "abc".into(),
        };
        let text = serde_json::to_string(&verdict).unwrap();
        assert_eq!(serde_json::from_str::<Verdict>(&text).unwrap(), verdict);
    }
}
