//! Diarisation de la parole en flux et transcription, en Rust, sans Python ni torch.
//!
//! Portage de [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper) :
//! ses chemins Nemotron gratuits et locaux, la diarisation Nemotron 3 en flux
//! puis la transcription Nemotron 3.5, chacune à travers son graphe ONNX.
//!
//! # Ce portage ne vise pas la vitesse
//!
//! C'est mesuré et non supposé. En chronométrant séparément l'appel au graphe et
//! le front-end, sur deux minutes d'audio : 99,8 % du temps est dans le graphe
//! ONNX, donc en C++ ; le front-end log-mel compte pour 0,1 % et la comptabilité
//! du cache pour 0,1 %. Le plafond d'un portage est donc **0,2 % du temps** —
//! un facteur temps réel de 0,754 au lieu de 0,755. La caisse `ort` appelle la
//! même bibliothèque C++.
//!
//! Il vise les endroits où Python ne va pas. Sur iOS et Android, on n'embarque
//! pas raisonnablement CPython et une pile scientifique : c'est un argument de
//! **possibilité** et non de performance. S'y ajoutent un binaire unique par
//! plateforme, aucune résolution de dépendances à l'installation, aucun
//! interpréteur à démarrer.
//!
//! # Ce qui est écrit, ce qui ne l'est pas
//!
//! Tout ce qui vit **autour** du graphe est là ; c'est l'essentiel du travail :
//! le front-end que le graphe attend à la trame près, la tenue du cache de
//! locuteurs et de la file, le calendrier et ses deux plafonds, enfin le choix du
//! moteur d'exécution. Rien de tout cela n'appelle le graphe, donc tout se
//! vérifie sans modèle — et tout est vérifié **contre le chemin Python**, parce
//! que les deux alimentent le même graphe et qu'un désaccord produirait
//! silencieusement d'autres probabilités.
//!
//! S'y ajoute l'hôte de la **transcription** Nemotron 3.5 ([`asr`]) : front-end
//! NeMo, normalisation, plan des fenêtres, décodage CTC et mots horodatés, avec
//! le lecteur des métadonnées du graphe ([`onnx_meta`]) qui sert aux deux
//! graphes sans charger leurs poids.
//!
//! # La liaison à ONNX Runtime, derrière la feature `onnx`
//!
//! `diar_session::NemotronDiar` et `asr_session::NemotronAsr` ouvrent les
//! graphes par la caisse `ort` et les font tourner de bout en bout : la
//! diarisation rend les mêmes probabilités que le jumeau Python à 2e-6 près, la
//! transcription le même texte à 4,4 % de WER près sur vingt minutes d'AMI en
//! int8, soit le bruit propre au graphe int8 (le jumeau Python contre lui-même,
//! optimisations du graphe coupées, s'écarte d'autant). Les graphes ne sont pas
//! distribués avec la caisse : on passe leur chemin.
//!
//! Les sessions s'ouvrent sur le moteur par défaut d'ONNX Runtime, le CPU. La
//! politique de [`accel`] (choisir le moteur sur une mesure) est écrite et
//! testée, mais pas encore branchée sur les sessions.
//!
//! Les chemins payants (Gladia, pyannoteAI) ne sont pas écrits.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod accel;
pub mod asr;
#[cfg(feature = "onnx")]
pub mod asr_session;
pub mod cache;
#[cfg(feature = "onnx")]
pub mod diar_session;
pub mod mel;
pub mod onnx_meta;
pub mod schedule;
pub mod segments;

pub use accel::{choose, holds_live, Provider, Timing, Verdict};
pub use cache::{
    boost_top_scores, compress_cache, disable_low_scores, log_pred_scores, streaming_update, Slot,
    StreamingState,
};
pub use mel::{log_mel, MelError, MelSettings};
pub use schedule::{Schedule, DEFAULT_LATENCY_S, MAX_SEQUENCE_FRAMES, STREAMING_MODES};
pub use segments::{segments_from_probabilities, Segment};
