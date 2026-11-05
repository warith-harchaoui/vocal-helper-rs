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
//! # Comment les sessions s'en servent (feature `onnx`)
//!
//! [`crate::asr_session::NemotronAsr`] et [`crate::diar_session::NemotronDiar`]
//! décident du moteur dans cet ordre d'autorité, le même que le jumeau Python :
//!
//! 1. l'argument `providers` de l'appelant ;
//! 2. sinon la variable [`PROVIDER_ENV`] (`VOCAL_HELPER_PROVIDER=cuda`, ou une
//!    liste ordonnée `tensorrt,cuda`) ;
//! 3. sinon la **mesure** : si plus d'un moteur est compilé dans la caisse
//!    (features `cuda`, `tensorrt`, `coreml`, `directml`, `openvino`), chacun
//!    fait un pas de chauffe puis une passe chronométrée sur [`PROBE_AUDIO`]
//!    d'audio synthétique ; [`choose`] retient le plus rapide et le [`Verdict`]
//!    est gardé sur disque sous l'empreinte du graphe ([`graph_digest`]), donc
//!    la mesure ne se paie qu'une fois par graphe et par machine. Avec le seul
//!    CPU compilé (la feature `onnx` seule), rien n'est mesuré.
//!
//! Un moteur qui échoue à s'enregistrer — pas de GPU, mauvaise version de CUDA,
//! feature absente à la compilation — est sauté avec un avertissement sur la
//! sortie d'erreur, puis le suivant est essayé ; le CPU ferme toujours la
//! marche. Ouvrir une session **n'échoue jamais** faute d'accélérateur. Seul
//! un nom inconnu dans [`PROVIDER_ENV`] est une erreur : une faute de frappe
//! ferait sinon croire qu'on tourne sur CUDA en tournant sur le CPU.

use std::fmt::Display;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// La variable d'environnement qui épingle le moteur, ou un ordre de moteurs.
///
/// Même nom et même rang que dans le jumeau Python : sous l'argument de
/// l'appelant, au-dessus de la mesure. Elle existe pour épingler une machine
/// sans toucher au code. Plusieurs noms séparés par des virgules forment un
/// ordre : `tensorrt,cuda` essaie TensorRT puis CUDA, puis le CPU.
pub const PROVIDER_ENV: &str = "VOCAL_HELPER_PROVIDER";

/// La variable d'environnement qui désigne le dossier des verdicts mesurés.
///
/// Par défaut `$XDG_CACHE_HOME/vocal-helper-rs`, sinon
/// `~/.cache/vocal-helper-rs` (`%LOCALAPPDATA%\vocal-helper-rs` sous Windows).
pub const CACHE_ENV: &str = "VOCAL_HELPER_RS_CACHE";

/// Ce qui empêche de décider d'un moteur.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AccelError {
    /// Un nom que rien ne désigne : une faute de frappe, refusée plutôt que
    /// devinée.
    #[error(
        "{PROVIDER_ENV} : « {0} » ne désigne aucun moteur d'exécution ; les noms acceptés sont \
         cpu, coreml, cuda, directml (ou dml), nnapi, openvino, qnn, tensorrt"
    )]
    UnknownName(String),
}

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

    /// Le moteur que désigne un nom écrit à la main, ou `None`.
    ///
    /// Tolérant sur la forme — casse, tirets, soulignés, suffixe
    /// `ExecutionProvider` — comme `provider_from_name` du jumeau Python, parce
    /// que ce nom s'écrit dans un shell, pas dans du code.
    ///
    /// ```
    /// use vocal_helper_rs::Provider;
    /// assert_eq!(Provider::from_name("cuda"), Some(Provider::Cuda));
    /// assert_eq!(Provider::from_name("CoreMLExecutionProvider"), Some(Provider::CoreMl));
    /// assert_eq!(Provider::from_name("metal"), None);
    /// ```
    pub fn from_name(name: &str) -> Option<Self> {
        let normalised: String = name
            .trim()
            .to_lowercase()
            .chars()
            .filter(|c| !matches!(c, '-' | '_' | ' '))
            .collect();
        let normalised = normalised.trim_end_matches("executionprovider");
        Some(match normalised {
            "cpu" => Self::Cpu,
            "coreml" => Self::CoreMl,
            "cuda" => Self::Cuda,
            "tensorrt" => Self::TensorRt,
            "directml" | "dml" => Self::DirectMl,
            "nnapi" => Self::Nnapi,
            "qnn" => Self::Qnn,
            "openvino" => Self::OpenVino,
            _ => return None,
        })
    }

    /// La feature de la caisse qui compile ce moteur, `None` pour le CPU.
    pub fn feature(self) -> Option<&'static str> {
        match self {
            Self::Cpu => None,
            Self::CoreMl => Some("coreml"),
            Self::Cuda => Some("cuda"),
            Self::TensorRt => Some("tensorrt"),
            Self::DirectMl => Some("directml"),
            Self::OpenVino => Some("openvino"),
            Self::Nnapi => Some("nnapi"),
            Self::Qnn => Some("qnn"),
        }
    }

    /// Ce moteur est-il compilé dans cette caisse ?
    ///
    /// Le CPU l'est toujours. Les autres le sont par leur feature ; NNAPI et QNN
    /// n'en ont pas encore, faute de les avoir essayés.
    pub fn is_compiled(self) -> bool {
        match self {
            Self::Cpu => true,
            Self::CoreMl => cfg!(feature = "coreml"),
            Self::Cuda => cfg!(feature = "cuda"),
            Self::TensorRt => cfg!(feature = "tensorrt"),
            Self::DirectMl => cfg!(feature = "directml"),
            Self::OpenVino => cfg!(feature = "openvino"),
            Self::Nnapi | Self::Qnn => false,
        }
    }

    /// Les candidats de [`Provider::candidates`] que cette compilation sait
    /// enregistrer : ce que la mesure essaie quand personne n'a rien demandé.
    pub fn compiled() -> Vec<Self> {
        Self::candidates()
            .into_iter()
            .filter(|p| p.is_compiled())
            .collect()
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
    /// Les moteurs qu'on a tenté de mesurer, y compris ceux qui ont échoué :
    /// une caisse recompilée avec un moteur de plus mérite une autre mesure.
    #[serde(default)]
    pub tried: Vec<Provider>,
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

    /// Le verdict vaut-il pour ce graphe **et** pour ces candidats ?
    ///
    /// Un verdict pris sur une compilation CPU seule ne dit rien de CUDA : si la
    /// liste des moteurs essayés a changé, on remesure.
    pub fn still_holds(&self, graph_digest: &str, candidates: &[Provider]) -> bool {
        let mut tried = self.tried.clone();
        let mut wanted = candidates.to_vec();
        tried.sort_by_key(|p| p.as_str());
        wanted.sort_by_key(|p| p.as_str());
        self.applies_to(graph_digest) && tried == wanted
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

/// La chauffe qui précède la passe chronométrée.
///
/// Le premier appel d'un moteur paie l'allocation, la compilation des noyaux
/// (CUDA, TensorRT) ou du modèle (CoreML) : le compter fausserait la mesure du
/// régime établi.
pub const PROBE_WARMUP: Duration = Duration::from_secs(5);

/// L'ordre explicite demandé, ou `None` si personne n'a rien dit.
///
/// L'argument de l'appelant d'abord, puis [`PROVIDER_ENV`]. Une liste vide
/// équivaut à ne rien demander.
///
/// # Erreurs
///
/// [`AccelError::UnknownName`] si la variable contient un nom inconnu.
pub fn requested_order(argument: Option<&[Provider]>) -> Result<Option<Vec<Provider>>, AccelError> {
    if let Some(order) = argument.filter(|o| !o.is_empty()) {
        return Ok(Some(order.to_vec()));
    }
    match std::env::var(PROVIDER_ENV) {
        Ok(raw) => parse_order(&raw),
        Err(_) => Ok(None),
    }
}

/// Lire un ordre écrit à la main : des noms séparés par des virgules.
///
/// ```
/// use vocal_helper_rs::{accel::parse_order, Provider};
/// assert_eq!(parse_order("tensorrt, cuda").unwrap(),
///            Some(vec![Provider::TensorRt, Provider::Cuda]));
/// assert_eq!(parse_order("  ").unwrap(), None);
/// assert!(parse_order("metal").is_err());
/// ```
///
/// # Erreurs
///
/// [`AccelError::UnknownName`] au premier nom inconnu.
pub fn parse_order(raw: &str) -> Result<Option<Vec<Provider>>, AccelError> {
    let mut order = Vec::new();
    for name in raw.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        let provider =
            Provider::from_name(name).ok_or_else(|| AccelError::UnknownName(name.to_owned()))?;
        if !order.contains(&provider) {
            order.push(provider);
        }
    }
    Ok(if order.is_empty() { None } else { Some(order) })
}

/// Ce qui sait enregistrer un moteur sur une session en construction.
///
/// La session ONNX Runtime en est un ; un enregistreur simulé en est un autre,
/// ce qui rend la politique de repli vérifiable sans GPU ni runtime.
pub trait Registrar {
    /// Enregistrer `provider`, ou dire pourquoi c'est impossible.
    ///
    /// # Erreurs
    ///
    /// Le message du runtime : bibliothèque CUDA introuvable, pas de pilote…
    fn register(&mut self, provider: Provider) -> Result<(), String>;

    /// Ce moteur peut-il seulement être essayé ? Par défaut, s'il est compilé
    /// dans la caisse ([`Provider::is_compiled`]) ; un enregistreur simulé peut
    /// prétendre le contraire, pour vérifier le repli sur toute plateforme.
    fn can_try(&self, provider: Provider) -> bool {
        provider.is_compiled()
    }
}

/// Ce que l'enregistrement a retenu, et ce qu'il a dû sauter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Registration {
    /// Les moteurs enregistrés, dans l'ordre ; le CPU ferme toujours la liste.
    pub active: Vec<Provider>,
    /// Un avertissement par moteur sauté, lisible tel quel.
    pub warnings: Vec<String>,
}

impl Registration {
    /// Le moteur de tête : celui qui prend les nœuds qu'il sait exécuter.
    pub fn lead(&self) -> Provider {
        self.active.first().copied().unwrap_or(Provider::Cpu)
    }
}

/// Enregistrer `order` dans l'ordre, sauter ce qui échoue, finir par le CPU.
///
/// Jamais d'échec : un moteur non compilé ou qui refuse de s'enregistrer
/// laisse un avertissement et passe la main au suivant. Le CPU ne s'enregistre
/// pas (ONNX Runtime l'ajoute de lui-même) mais il est toujours compté en
/// dernier ; placé plus tôt dans l'ordre, il clôt la liste, puisqu'il exécute
/// tous les opérateurs et ne laisserait rien aux suivants.
///
/// ```
/// use vocal_helper_rs::accel::{register_in_order, Registrar};
/// use vocal_helper_rs::Provider;
///
/// struct SansGpu;
/// impl Registrar for SansGpu {
///     fn register(&mut self, p: Provider) -> Result<(), String> {
///         Err(format!("{} : libcudart.so.13 introuvable", p.as_str()))
///     }
/// }
/// let r = register_in_order(&[Provider::Cuda], &mut SansGpu);
/// assert_eq!(r.active, vec![Provider::Cpu]);
/// assert_eq!(r.warnings.len(), 1);
/// ```
pub fn register_in_order<R: Registrar + ?Sized>(
    order: &[Provider],
    registrar: &mut R,
) -> Registration {
    let mut registration = Registration::default();
    for &provider in order {
        if provider == Provider::Cpu {
            break;
        }
        if registration.active.contains(&provider) {
            continue;
        }
        if !registrar.can_try(provider) {
            let why = match provider.feature() {
                Some(f) if matches!(provider, Provider::Nnapi | Provider::Qnn) => {
                    format!("aucune feature `{f}` n'existe encore dans cette caisse")
                }
                Some(f) => format!("caisse compilée sans la feature `{f}`"),
                None => String::new(),
            };
            registration.warnings.push(format!(
                "{} sauté ({why}) ; repli sur le suivant",
                provider.as_str()
            ));
            continue;
        }
        match registrar.register(provider) {
            Ok(()) => registration.active.push(provider),
            Err(e) => registration.warnings.push(format!(
                "{} n'a pas pu s'enregistrer ({e}) ; repli sur le suivant",
                provider.as_str()
            )),
        }
    }
    registration.active.push(Provider::Cpu);
    registration
}

/// Mesurer chaque candidat sur l'appareil, garder la session du plus rapide.
///
/// `open` ouvre une session dont `provider` mène, ou rend `Ok(None)` s'il n'a
/// pas pu s'enregistrer (la session serait alors un CPU déguisé : la mesurer
/// sous son nom mentirait). `time` fait la chauffe puis rend la durée de la
/// passe chronométrée sur `audio_s` secondes d'audio. Un moteur qui échoue est
/// noté dans les avertissements et laissé de côté, jamais fatal.
///
/// Rend les mesures, le moteur gagnant avec sa session, et les avertissements.
pub fn measure<S, E: Display>(
    candidates: &[Provider],
    audio_s: f64,
    mut open: impl FnMut(Provider) -> Result<Option<S>, E>,
    mut time: impl FnMut(&mut S) -> Result<Duration, E>,
) -> (Vec<Timing>, Option<(Provider, S)>, Vec<String>) {
    let mut timings = Vec::new();
    let mut warnings = Vec::new();
    let mut best: Option<(Provider, S, f64)> = None;
    for &provider in candidates {
        let mut session = match open(provider) {
            Ok(Some(session)) => session,
            Ok(None) => continue,
            Err(e) => {
                warnings.push(format!(
                    "{} : ouverture impossible ({e})",
                    provider.as_str()
                ));
                continue;
            }
        };
        let elapsed = match time(&mut session) {
            Ok(elapsed) => elapsed,
            Err(e) => {
                warnings.push(format!(
                    "{} : inférence impossible ({e})",
                    provider.as_str()
                ));
                continue;
            }
        };
        let rtf = elapsed.as_secs_f64() / audio_s;
        timings.push(Timing {
            provider,
            real_time_factor: rtf,
        });
        if choose(&timings) == Some(provider) {
            best = Some((provider, session, rtf));
        }
    }
    (timings, best.map(|(p, s, _)| (p, s)), warnings)
}

/// L'empreinte d'un graphe sur disque, sans le lire en entier.
///
/// Un graphe int8 pèse 669 Mo : le hacher en entier à chaque ouverture coûterait
/// plus que la session. On hache donc sa taille, son premier et son dernier
/// mégaoctet (FNV-1a 64 bits, stable d'une version de Rust à l'autre, ce que
/// `DefaultHasher` ne promet pas). Deux exports différents d'un même modèle
/// diffèrent dès l'en-tête ; c'est ce qu'il faut pour qu'un verdict ne serve
/// pas un autre graphe.
///
/// # Erreurs
///
/// Une erreur d'entrée-sortie si le fichier est illisible.
pub fn graph_digest(path: &Path) -> io::Result<String> {
    const SPAN: u64 = 1 << 20;
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    feed(&len.to_le_bytes());
    let mut buffer = Vec::new();
    file.by_ref().take(SPAN).read_to_end(&mut buffer)?;
    feed(&buffer);
    if len > SPAN {
        buffer.clear();
        file.seek(SeekFrom::Start(len.saturating_sub(SPAN).max(SPAN)))?;
        file.read_to_end(&mut buffer)?;
        feed(&buffer);
    }
    Ok(format!("fnv1a64-{len}-{hash:016x}"))
}

/// Le dossier où les verdicts sont gardés, s'il y en a un.
pub fn verdict_dir() -> Option<PathBuf> {
    let var = |name: &str| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if let Some(dir) = var(CACHE_ENV) {
        return Some(dir);
    }
    if let Some(dir) = var("XDG_CACHE_HOME") {
        return Some(dir.join("vocal-helper-rs"));
    }
    if cfg!(windows) {
        if let Some(dir) = var("LOCALAPPDATA") {
            return Some(dir.join("vocal-helper-rs"));
        }
    }
    var("HOME").map(|home| home.join(".cache").join("vocal-helper-rs"))
}

fn verdict_path(dir: &Path, graph_digest: &str) -> PathBuf {
    dir.join(format!("accel-{graph_digest}.json"))
}

/// Relire le verdict gardé pour ce graphe, s'il existe et se lit.
pub fn load_verdict(dir: &Path, graph_digest: &str) -> Option<Verdict> {
    let text = std::fs::read_to_string(verdict_path(dir, graph_digest)).ok()?;
    serde_json::from_str::<Verdict>(&text)
        .ok()
        .filter(|v| v.applies_to(graph_digest))
}

/// Garder un verdict pour la prochaine ouverture du même graphe.
///
/// # Erreurs
///
/// Une erreur d'entrée-sortie si le dossier ne s'écrit pas.
pub fn save_verdict(dir: &Path, verdict: &Verdict) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let text = serde_json::to_string_pretty(verdict).map_err(io::Error::other)?;
    std::fs::write(verdict_path(dir, &verdict.graph_digest), text)
}

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
            tried: vec![],
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
            tried: vec![],
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
            tried: vec![],
        };
        let text = serde_json::to_string(&verdict).unwrap();
        assert_eq!(serde_json::from_str::<Verdict>(&text).unwrap(), verdict);
    }

    // ----- enregistrement et repli, avec un enregistreur simulé ----------

    /// Un enregistreur qui accepte ce qu'on lui dit d'accepter, et note tout.
    struct Simule {
        accepte: Vec<Provider>,
        essais: Vec<Provider>,
    }

    /// Le simulé fait comme si tous les moteurs à feature étaient compilés :
    /// le repli se vérifie ainsi sur toute machine, sans GPU ni runtime.
    impl Registrar for Simule {
        fn can_try(&self, provider: Provider) -> bool {
            !matches!(provider, Provider::Nnapi | Provider::Qnn)
        }

        fn register(&mut self, provider: Provider) -> Result<(), String> {
            self.essais.push(provider);
            if self.accepte.contains(&provider) {
                Ok(())
            } else {
                Err("libcudart.so.13: cannot open shared object file".into())
            }
        }
    }

    fn simule(accepte: &[Provider]) -> Simule {
        Simule {
            accepte: accepte.to_vec(),
            essais: vec![],
        }
    }

    #[test]
    fn le_cpu_ferme_toujours_la_liste() {
        let r = register_in_order(&[], &mut simule(&[]));
        assert_eq!(r.active, vec![Provider::Cpu]);
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn un_moteur_qui_refuse_de_s_enregistrer_laisse_un_avertissement_pas_une_erreur() {
        // Le cas d'une machine sans GPU, ou avec CUDA 12 face à un runtime CUDA 13.
        let mut registrar = simule(&[]);
        let order = vec![Provider::TensorRt, Provider::Cuda];
        let r = register_in_order(&order, &mut registrar);
        assert_eq!(r.active, vec![Provider::Cpu]);
        assert_eq!(r.warnings.len(), order.len());
        assert_eq!(registrar.essais, order);
        for w in &r.warnings {
            assert!(w.contains("repli"), "{w}");
        }
    }

    #[test]
    fn hors_simulation_un_moteur_sans_sa_feature_n_est_jamais_essaye() {
        // L'enregistreur par défaut consulte `is_compiled` : sans la feature
        // `cuda`, CUDA est sauté avec un avertissement qui nomme la feature.
        struct Reel(Vec<Provider>);
        impl Registrar for Reel {
            fn register(&mut self, p: Provider) -> Result<(), String> {
                self.0.push(p);
                Ok(())
            }
        }
        let mut registrar = Reel(vec![]);
        let r = register_in_order(&[Provider::Cuda], &mut registrar);
        if Provider::Cuda.is_compiled() {
            assert_eq!(r.active, vec![Provider::Cuda, Provider::Cpu]);
        } else {
            assert!(registrar.0.is_empty());
            assert_eq!(r.active, vec![Provider::Cpu]);
            assert!(
                r.warnings[0].contains("feature `cuda`"),
                "{}",
                r.warnings[0]
            );
        }
    }

    #[test]
    fn un_moteur_non_compile_est_saute_sans_meme_etre_essaye() {
        // NNAPI n'a pas de feature : l'enregistreur ne doit jamais le voir.
        let mut registrar = simule(&[Provider::Nnapi]);
        let r = register_in_order(&[Provider::Nnapi], &mut registrar);
        assert_eq!(r.active, vec![Provider::Cpu]);
        assert!(registrar.essais.is_empty());
        assert!(r.warnings[0].contains("NnapiExecutionProvider"));
    }

    #[test]
    fn le_premier_qui_s_enregistre_mene_et_le_suivant_reste_en_reserve() {
        // TensorRT refuse, CUDA accepte, CoreML accepte aussi : CUDA mène,
        // CoreML reste en réserve, le CPU ferme la marche.
        let mut registrar = simule(&[Provider::Cuda, Provider::CoreMl]);
        let r = register_in_order(
            &[Provider::TensorRt, Provider::Cuda, Provider::CoreMl],
            &mut registrar,
        );
        assert_eq!(
            r.active,
            vec![Provider::Cuda, Provider::CoreMl, Provider::Cpu]
        );
        assert_eq!(r.lead(), Provider::Cuda);
        assert_eq!(r.warnings.len(), 1);
        assert!(r.warnings[0].starts_with("TensorrtExecutionProvider"));
    }

    #[test]
    fn le_cpu_demande_en_tete_clot_la_liste() {
        // Il exécute tous les opérateurs : un moteur après lui ne recevrait rien.
        let mut registrar = simule(&Provider::candidates());
        let r = register_in_order(&[Provider::Cpu, Provider::Cuda], &mut registrar);
        assert_eq!(r.active, vec![Provider::Cpu]);
        assert!(registrar.essais.is_empty());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn avec_cuda_compile_un_echec_d_enregistrement_retombe_sur_le_cpu() {
        let mut registrar = simule(&[]);
        let r = register_in_order(&[Provider::Cuda], &mut registrar);
        assert_eq!(registrar.essais, vec![Provider::Cuda]);
        assert_eq!(r.lead(), Provider::Cpu);
        assert!(r.warnings[0].contains("CUDAExecutionProvider"));
    }

    // ----- la mesure, avec des durées simulées ----------------------------

    #[test]
    fn la_mesure_garde_la_session_du_plus_rapide() {
        let durees = |p: Provider| match p {
            Provider::Cuda => 0.9,
            Provider::Cpu => 15.8,
            _ => 37.7,
        };
        let candidats = [Provider::Cuda, Provider::Cpu, Provider::CoreMl];
        let (timings, best, warnings) = measure(
            &candidats,
            30.0,
            |p| Ok::<_, String>(Some(p)),
            |p: &mut Provider| Ok(Duration::from_secs_f64(durees(*p))),
        );
        assert_eq!(timings.len(), 3);
        assert_eq!(best, Some((Provider::Cuda, Provider::Cuda)));
        assert!(warnings.is_empty());
        assert!((timings[1].real_time_factor - 15.8 / 30.0).abs() < 1e-12);
    }

    #[test]
    fn un_moteur_qui_ne_s_ouvre_pas_ou_plante_n_est_pas_mesure() {
        let (timings, best, warnings) = measure(
            &[Provider::TensorRt, Provider::Cuda, Provider::Cpu],
            30.0,
            |p| match p {
                // Retombé sur le CPU à l'enregistrement : pas de mesure sous ce nom.
                Provider::TensorRt => Ok(None),
                _ => Ok::<_, String>(Some(p)),
            },
            |p: &mut Provider| match p {
                Provider::Cuda => Err("CUDA failure 35: driver version is insufficient".into()),
                _ => Ok(Duration::from_secs(15)),
            },
        );
        assert_eq!(
            timings,
            vec![Timing {
                provider: Provider::Cpu,
                real_time_factor: 0.5
            }]
        );
        assert_eq!(best.map(|(p, _)| p), Some(Provider::Cpu));
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("CUDAExecutionProvider"));
    }

    #[test]
    fn sans_aucun_moteur_mesurable_on_ne_rend_rien() {
        let (timings, best, _) = measure(
            &[Provider::Cuda],
            30.0,
            |_| Ok::<Option<()>, String>(None),
            |_: &mut ()| Ok(Duration::ZERO),
        );
        assert!(timings.is_empty());
        assert!(best.is_none());
    }

    // ----- noms, ordre explicite ------------------------------------------

    #[test]
    fn les_noms_du_jumeau_python_sont_tous_reconnus() {
        for (name, provider) in [
            ("cpu", Provider::Cpu),
            ("coreml", Provider::CoreMl),
            ("cuda", Provider::Cuda),
            ("tensorrt", Provider::TensorRt),
            ("directml", Provider::DirectMl),
            ("dml", Provider::DirectMl),
            ("nnapi", Provider::Nnapi),
            ("qnn", Provider::Qnn),
            ("openvino", Provider::OpenVino),
        ] {
            assert_eq!(Provider::from_name(name), Some(provider), "{name}");
            assert_eq!(Provider::from_name(provider.as_str()), Some(provider));
        }
        assert_eq!(Provider::from_name(" Tensor-RT "), Some(Provider::TensorRt));
    }

    #[test]
    fn un_nom_inconnu_est_une_erreur_pas_un_repli() {
        assert_eq!(
            parse_order("cuda,metal"),
            Err(AccelError::UnknownName("metal".into()))
        );
    }

    #[test]
    fn l_argument_passe_avant_l_environnement() {
        assert_eq!(
            requested_order(Some(&[Provider::Cuda])),
            Ok(Some(vec![Provider::Cuda]))
        );
    }

    #[test]
    fn un_ordre_ecrit_deux_fois_ne_compte_qu_une() {
        assert_eq!(
            parse_order("cuda,CUDA,cpu"),
            Ok(Some(vec![Provider::Cuda, Provider::Cpu]))
        );
    }

    // ----- verdict gardé ---------------------------------------------------

    #[test]
    fn un_verdict_ne_tient_plus_si_les_candidats_ont_change() {
        // Une caisse recompilée avec `cuda` doit remesurer.
        let verdict = Verdict {
            chosen: Provider::Cpu,
            timings: vec![timing(Provider::Cpu, 0.5)],
            graph_digest: "abc".into(),
            tried: vec![Provider::Cpu],
        };
        assert!(verdict.still_holds("abc", &[Provider::Cpu]));
        assert!(!verdict.still_holds("abc", &[Provider::Cuda, Provider::Cpu]));
        assert!(!verdict.still_holds("def", &[Provider::Cpu]));
    }

    #[test]
    fn un_verdict_de_la_0_1_se_relit_encore() {
        // Écrit avant le champ `tried` : il se relit, et sera remesuré.
        let text = r#"{"chosen":"Cpu","timings":[],"graph_digest":"abc"}"#;
        let verdict: Verdict = serde_json::from_str(text).unwrap();
        assert!(verdict.tried.is_empty());
        assert!(!verdict.still_holds("abc", &[Provider::Cpu]));
    }

    fn dossier_temporaire(nom: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("vocal-helper-rs-{nom}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn un_verdict_survit_au_disque_et_reste_attache_a_son_graphe() {
        let dir = dossier_temporaire("verdict");
        let verdict = Verdict {
            chosen: Provider::Cuda,
            timings: vec![timing(Provider::Cuda, 0.03), timing(Provider::Cpu, 0.5)],
            graph_digest: "fnv1a64-1-00".into(),
            tried: vec![Provider::Cuda, Provider::Cpu],
        };
        save_verdict(&dir, &verdict).unwrap();
        assert_eq!(load_verdict(&dir, "fnv1a64-1-00"), Some(verdict));
        assert_eq!(load_verdict(&dir, "fnv1a64-2-00"), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn l_empreinte_suit_le_contenu_et_non_le_nom() {
        let dir = dossier_temporaire("empreinte");
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b, c) = (dir.join("a.onnx"), dir.join("b.onnx"), dir.join("c.onnx"));
        // Plus de deux mégaoctets, pour passer par la tête et la queue.
        let mut content = vec![7u8; (1 << 21) + 10];
        std::fs::write(&a, &content).unwrap();
        std::fs::write(&b, &content).unwrap();
        *content.last_mut().unwrap() = 8;
        std::fs::write(&c, &content).unwrap();
        let (da, db, dc) = (
            graph_digest(&a).unwrap(),
            graph_digest(&b).unwrap(),
            graph_digest(&c).unwrap(),
        );
        assert_eq!(da, db);
        assert_ne!(da, dc, "la queue du fichier doit compter");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
