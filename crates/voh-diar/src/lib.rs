//! Diarisation en flux par Nemotron 3, sans torch.
//!
//! Le graphe ONNX est **un pas de flux**, pas un diariseur de fichier : il
//! reçoit `[cache | file | morceau]` et rend des probabilités. Cette caisse
//! porte tout ce qui vit autour de lui — le calendrier et ses deux plafonds, la
//! tenue du cache de locuteurs et de la file, et la traduction des probabilités
//! en tours de parole.
//!
//! Rien ici n'appelle le graphe : c'est de l'arithmétique sur des tableaux, donc
//! entièrement vérifiable sans modèle. La liaison à ONNX Runtime viendra dans un
//! module à part, derrière un drapeau, pour que ce raisonnement reste testable
//! sur une machine qui n'a pas téléchargé le runtime.
//!
//! La parité avec le chemin Python est vérifiée dans `tests/parite_python.rs` :
//! les deux alimentent le même graphe, donc un désaccord produirait
//! silencieusement d'autres probabilités.

pub mod cache;
pub mod schedule;
pub mod segments;

pub use cache::{
    boost_top_scores, compress_cache, disable_low_scores, log_pred_scores, streaming_update, Slot,
    StreamingState,
};
pub use schedule::{Schedule, DEFAULT_LATENCY_S, MAX_SEQUENCE_FRAMES, STREAMING_MODES};
pub use segments::{segments_from_probabilities, Segment};
