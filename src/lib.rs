//! Diarisation de la parole en flux, en Rust, sans Python ni torch.
//!
//! Portage de [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper),
//! et plus précisément de son chemin de diarisation gratuit et local : le modèle
//! Nemotron 3 Diarization à travers son graphe ONNX.
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
//! plateforme, aucune résolution de dépendances à l'installation, et aucun
//! interpréteur à démarrer.
//!
//! # Ce qui est écrit, et ce qui ne l'est pas
//!
//! Tout ce qui vit **autour** du graphe est là, et c'est l'essentiel du travail :
//! le front-end que le graphe attend à la trame près, la tenue du cache de
//! locuteurs et de la file, le calendrier et ses deux plafonds, et le choix du
//! moteur d'exécution. Rien de tout cela n'appelle le graphe, donc tout se
//! vérifie sans modèle — et tout est vérifié **contre le chemin Python**, parce
//! que les deux alimentent le même graphe et qu'un désaccord produirait
//! silencieusement d'autres probabilités.
//!
//! La liaison à ONNX Runtime et les chemins payants ne sont pas encore écrits.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod accel;
pub mod cache;
pub mod mel;
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
