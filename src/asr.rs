//! L'hôte de la transcription Nemotron 3.5 : tout ce qui entoure le graphe ONNX.
//!
//! Le graphe (`nemotron-asr-ctc`, exporté côté Python par
//! `studies/nemotron_asr_export.py`) prend un log-mel normalisé `(1, 128, T)` et
//! rend, toutes les 40 ms, le jeton le plus probable, sa log-probabilité et
//! celle du blanc. Ce module est le portage de `vocal_helper.nemotron_asr` :
//!
//! - le front-end de NeMo, avec la fenêtre et le banc de filtres que le graphe
//!   porte dans ses métadonnées — remplissage **constant** plutôt que par réflexion
//!   comme le front-end du diariseur ([`crate::mel`]) ;
//! - la normalisation par bande ;
//! - le plan des fenêtres de 30 s ;
//! - le décodage CTC glouton et la reconstruction des mots horodatés ; les
//!   balises (`<spk:N>`, `<ev:…>`, `<sty:…>`) sortent à part.
//!
//! Rien ici n'appelle le graphe : tout se vérifie contre le Python sur fixtures.

use std::collections::HashMap;

use realfft::RealFftPlanner;
use thiserror::Error;

use crate::mel::MelError;
use crate::onnx_meta::{base64_f32le, MetaError};

/// Préfixe des métadonnées du graphe ASR.
pub const META_PREFIX: &str = "vocal_helper.asr.";
/// Le contrat que cet hôte sait lire.
pub const CONTRACT: &str = "nemotron-asr-ctc-1";
/// Le séparateur de mots de SentencePiece.
pub const WORD_START: char = '\u{2581}';
/// Le plancher d'écart-type de NeMo (`features.CONSTANT`).
pub const STD_FLOOR: f64 = 1e-5;

/// Un graphe dont les métadonnées ne conviennent pas.
#[derive(Debug, Error)]
pub enum AsrError {
    /// Une clé attendue manque.
    #[error("métadonnée {0} absente du graphe")]
    Missing(String),
    /// Une valeur ne se lit pas.
    #[error("métadonnée {key} illisible : {reason}")]
    Invalid {
        /// La clé fautive.
        key: String,
        /// Pourquoi.
        reason: String,
    },
    /// Le graphe suit un autre contrat.
    #[error("contrat {found:?}, cet hôte lit {CONTRACT:?}")]
    Contract {
        /// Le contrat trouvé.
        found: String,
    },
    /// Décodage des flottants.
    #[error(transparent)]
    Meta(#[from] MetaError),
}

/// Le front-end et le vocabulaire, tels que le graphe les porte.
#[derive(Debug, Clone)]
pub struct AsrSettings {
    /// Banc de filtres `(n_mels, n_fft / 2 + 1)`, ligne par ligne.
    pub fb: Vec<f32>,
    /// Nombre de bandes.
    pub n_mels: usize,
    /// Fenêtre d'analyse, centrée dans la transformée.
    pub window: Vec<f32>,
    /// Les pièces SentencePiece, le blanc exclu.
    pub vocab: Vec<String>,
    /// L'indice du blanc CTC.
    pub blank_id: u32,
    /// 16 kHz.
    pub sample_rate: u32,
    /// Taille de la transformée.
    pub n_fft: usize,
    /// Avance entre trames mel (10 ms).
    pub hop_length: usize,
    /// Pré-accentuation.
    pub preemphasis: f32,
    /// Plancher du logarithme.
    pub log_guard: f32,
    /// Trames mel empilées par trame d'encodeur.
    pub stack: usize,
    /// Durée d'une trame CTC, en secondes.
    pub frame_s: f64,
    /// Longueur d'une fenêtre de l'encodeur, en secondes.
    pub chunk_s: f64,
}

impl AsrSettings {
    /// Lire les réglages depuis les métadonnées du graphe.
    ///
    /// # Erreurs
    ///
    /// [`AsrError`] si une clé manque ou ne se lit pas, voire si le contrat diffère.
    pub fn from_metadata(meta: &HashMap<String, String>) -> Result<Self, AsrError> {
        let get = |key: &str| -> Result<&str, AsrError> {
            meta.get(&format!("{META_PREFIX}{key}"))
                .map(String::as_str)
                .ok_or_else(|| AsrError::Missing(format!("{META_PREFIX}{key}")))
        };
        fn parse<T: std::str::FromStr>(key: &str, raw: &str) -> Result<T, AsrError> {
            raw.trim().parse().map_err(|_| AsrError::Invalid {
                key: key.into(),
                reason: format!("{raw:?}"),
            })
        }
        let contract = get("contract")?;
        if contract != CONTRACT {
            return Err(AsrError::Contract {
                found: contract.into(),
            });
        }
        let n_mels: usize = parse("n_mels", get("n_mels")?)?;
        let n_fft: usize = parse("n_fft", get("n_fft")?)?;
        let fb = base64_f32le(get("mel_fb")?)?;
        if fb.len() != n_mels * (n_fft / 2 + 1) {
            return Err(AsrError::Invalid {
                key: "mel_fb".into(),
                reason: format!("{} valeurs pour {n_mels} × {}", fb.len(), n_fft / 2 + 1),
            });
        }
        let vocab: Vec<String> =
            serde_json::from_str(get("vocab")?).map_err(|e| AsrError::Invalid {
                key: "vocab".into(),
                reason: e.to_string(),
            })?;
        Ok(Self {
            fb,
            n_mels,
            window: base64_f32le(get("mel_window")?)?,
            vocab,
            blank_id: parse("blank_id", get("blank_id")?)?,
            sample_rate: parse("sample_rate", get("sample_rate")?)?,
            n_fft,
            hop_length: parse("hop_length", get("hop_length")?)?,
            preemphasis: parse("preemphasis", get("preemphasis")?)?,
            log_guard: parse::<f64>("log_guard", get("log_guard")?)? as f32,
            stack: parse("stack", get("stack")?)?,
            frame_s: parse("frame_s", get("frame_s")?)?,
            chunk_s: parse("chunk_s", get("chunk_s")?)?,
        })
    }

    /// Trames mel par trame CTC (4 : 40 ms sur 10 ms).
    pub fn mel_per_frame(&self) -> usize {
        (self.frame_s * f64::from(self.sample_rate) / self.hop_length as f64).round() as usize
    }
}

/// Le log-mel de NeMo, `n_mels` bandes de `T` trames, avec le nombre de trames valides.
///
/// # Erreurs
///
/// [`MelError::SampleRate`] si la fréquence n'est pas celle du modèle.
pub fn log_mel(
    audio: &[f32],
    sample_rate: u32,
    s: &AsrSettings,
) -> Result<(Vec<Vec<f32>>, usize), MelError> {
    if sample_rate != s.sample_rate {
        return Err(MelError::SampleRate {
            expected: s.sample_rate,
            received: sample_rate,
        });
    }
    let valid = audio.len() / s.hop_length;
    if audio.is_empty() {
        return Ok((vec![vec![0.0; 1]; s.n_mels], 0));
    }
    let half = s.n_fft / 2;
    let mut padded = vec![0.0f32; audio.len() + 2 * half];
    padded[half] = audio[0];
    for i in 1..audio.len() {
        padded[half + i] = audio[i] - s.preemphasis * audio[i - 1];
    }
    let mut window = vec![0.0f32; s.n_fft];
    let offset = (s.n_fft - s.window.len()) / 2;
    window[offset..offset + s.window.len()].copy_from_slice(&s.window);

    let n_bins = s.n_fft / 2 + 1;
    let frames = 1 + (padded.len() - s.n_fft) / s.hop_length;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(s.n_fft);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    let mut mel = vec![vec![0.0f32; frames]; s.n_mels];
    let mut power = vec![0.0f32; n_bins];
    for t in 0..frames.min(valid) {
        let start = t * s.hop_length;
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = f64::from(padded[start + i] * window[i]);
        }
        fft.process(&mut input, &mut output)
            .expect("les longueurs viennent du plan, donc elles concordent");
        for (p, c) in power.iter_mut().zip(output.iter()) {
            *p = c.norm_sqr() as f32;
        }
        for (band, row) in mel.iter_mut().enumerate() {
            let filter = &s.fb[band * n_bins..(band + 1) * n_bins];
            let energy: f32 = filter.iter().zip(power.iter()).map(|(&w, &p)| w * p).sum();
            row[t] = (energy + s.log_guard).ln();
        }
    }
    Ok((mel, valid))
}

/// Centrer et réduire chaque bande sur les trames valides, comme NeMo.
///
/// Écart-type sans biais, plancher ajouté après, trames invalides à zéro.
pub fn normalize_per_feature(mel: &mut [Vec<f32>], valid: usize) {
    for row in mel.iter_mut() {
        let core = &row[..valid.min(row.len())];
        let n = core.len() as f64;
        let mean = if n > 0.0 {
            core.iter().map(|&v| f64::from(v)).sum::<f64>() / n
        } else {
            0.0
        };
        let std = if n > 1.0 {
            (core
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / (n - 1.0))
                .sqrt()
        } else {
            0.0
        };
        for (t, v) in row.iter_mut().enumerate() {
            *v = if t < valid {
                ((f64::from(*v) - mean) / (std + STD_FLOOR)) as f32
            } else {
                0.0
            };
        }
    }
}

/// Une fenêtre à envoyer au graphe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// Première trame mel.
    pub start: usize,
    /// Trames mel valides.
    pub len: usize,
    /// Trames mel envoyées, complétées à un multiple de `stack`.
    pub padded: usize,
    /// L'entrée `enc_len` du graphe.
    pub enc_len: i64,
    /// Trames CTC à garder en sortie.
    pub keep: usize,
}

/// Le découpage en fenêtres indépendantes de `chunk_s`, comme l'encodeur de référence.
///
/// ```
/// # use vocal_helper_rs::asr::plan_windows;
/// let w = plan_windows(6500, 3000, 8, 4);
/// assert_eq!(w.iter().map(|w| w.padded).collect::<Vec<_>>(), vec![3000, 3000, 504]);
/// assert_eq!(w.iter().map(|w| w.keep).sum::<usize>(), 1625);
/// ```
pub fn plan_windows(valid: usize, chunk: usize, stack: usize, mel_per_frame: usize) -> Vec<Window> {
    (0..valid)
        .step_by(chunk.max(1))
        .map(|start| {
            let len = chunk.min(valid - start);
            let padded = len.div_ceil(stack) * stack;
            Window {
                start,
                len,
                padded,
                enc_len: (padded / stack) as i64,
                keep: len.div_ceil(mel_per_frame),
            }
        })
        .collect()
}

/// Un jeton émis par le décodage glouton, avec ses trames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CtcToken {
    /// Indice dans le vocabulaire.
    pub id: u32,
    /// Première trame.
    pub first: usize,
    /// Dernière trame.
    pub last: usize,
    /// Meilleure log-probabilité sur ses trames.
    pub logprob: f32,
}

/// Fusionner les répétitions et retirer les blancs, en gardant les trames.
///
/// ```
/// # use vocal_helper_rs::asr::greedy_ctc;
/// let t = greedy_ctc(&[9, 5, 5, 9, 5, 7], &[0.0; 6], 9);
/// let spans: Vec<_> = t.iter().map(|t| (t.id, t.first, t.last)).collect();
/// assert_eq!(spans, vec![(5, 1, 2), (5, 4, 4), (7, 5, 5)]);
/// ```
pub fn greedy_ctc(ids: &[i32], logprob: &[f32], blank_id: u32) -> Vec<CtcToken> {
    let mut out: Vec<CtcToken> = Vec::new();
    let mut prev = blank_id as i64;
    for (frame, (&id, &lp)) in ids.iter().zip(logprob.iter()).enumerate() {
        let id64 = i64::from(id);
        if id64 != i64::from(blank_id) {
            match out.last_mut() {
                Some(last) if id64 == prev => {
                    last.last = frame;
                    last.logprob = last.logprob.max(lp);
                }
                _ => out.push(CtcToken {
                    id: id as u32,
                    first: frame,
                    last: frame,
                    logprob: lp,
                }),
            }
        }
        prev = id64;
    }
    out
}

/// Un mot horodaté.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Word {
    /// Le texte du mot.
    pub text: String,
    /// Début, en secondes.
    pub start_s: f64,
    /// Fin, en secondes.
    pub end_s: f64,
    /// Moyenne des probabilités de ses pièces.
    pub confidence: f64,
}

/// Une balise émise par la tête CTC : locuteur, événement ou style.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tag {
    /// Instant, en secondes.
    pub at_s: f64,
    /// La balise, `<spk:0>` par exemple.
    pub tag: String,
}

fn is_tag(piece: &str) -> bool {
    piece.len() > 2 && piece.starts_with('<') && piece.ends_with('>')
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Regrouper les pièces en mots horodatés puis sortir les balises à part.
///
/// Une pièce qui commence par `▁` ouvre un mot, les autres le prolongent, une
/// balise clôt le mot en cours. `<unk>` est avalé.
pub fn tokens_to_words(
    tokens: &[CtcToken],
    vocab: &[String],
    frame_s: f64,
) -> (Vec<Word>, Vec<Tag>) {
    let mut words = Vec::new();
    let mut tags = Vec::new();
    let mut text = String::new();
    let (mut first, mut last) = (0usize, 0usize);
    let mut probs: Vec<f64> = Vec::new();

    let close = |text: &str, first: usize, last: usize, probs: &[f64], words: &mut Vec<Word>| {
        if !text.is_empty() {
            let confidence = probs.iter().map(|p| p.exp()).sum::<f64>() / probs.len().max(1) as f64;
            words.push(Word {
                text: text.to_owned(),
                start_s: round3(first as f64 * frame_s),
                end_s: round3((last + 1) as f64 * frame_s),
                confidence,
            });
        }
    };

    for token in tokens {
        let piece = vocab
            .get(token.id as usize)
            .map(String::as_str)
            .unwrap_or("");
        if is_tag(piece) {
            close(&text, first, last, &probs, &mut words);
            text.clear();
            probs.clear();
            if piece != "<unk>" {
                tags.push(Tag {
                    at_s: round3(token.first as f64 * frame_s),
                    tag: piece.to_owned(),
                });
            }
            continue;
        }
        if piece.starts_with(WORD_START) || text.is_empty() {
            close(&text, first, last, &probs, &mut words);
            text = piece.trim_start_matches(WORD_START).to_owned();
            first = token.first;
            probs.clear();
            if text.is_empty() {
                continue;
            }
        } else {
            text.push_str(piece);
        }
        last = token.last;
        probs.push(f64::from(token.logprob));
    }
    close(&text, first, last, &probs, &mut words);
    (words, tags)
}

/// La probabilité de parole par trame, `1 − exp(blanc)`.
pub fn speech_probability(blank_logprob: &[f32]) -> Vec<f32> {
    blank_logprob.iter().map(|&b| 1.0 - b.exp()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vec<String> {
        [
            "<spk:1>",
            "\u{2581}bon",
            "jour",
            "\u{2581}à",
            "\u{2581}tous",
            "<unk>",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn tok(id: u32, first: usize, last: usize) -> CtcToken {
        CtcToken {
            id,
            first,
            last,
            logprob: 0.0,
        }
    }

    #[test]
    fn les_mots_et_les_balises_se_separent() {
        let toks = [
            tok(1, 0, 1),
            tok(2, 3, 3),
            tok(0, 5, 5),
            tok(3, 6, 6),
            tok(4, 8, 9),
            tok(5, 10, 10),
        ];
        let (words, tags) = tokens_to_words(&toks, &vocab(), 0.04);
        let got: Vec<_> = words
            .iter()
            .map(|w| (w.text.as_str(), w.start_s, w.end_s))
            .collect();
        assert_eq!(
            got,
            vec![
                ("bonjour", 0.0, 0.16),
                ("à", 0.24, 0.28),
                ("tous", 0.32, 0.4)
            ]
        );
        assert_eq!(
            tags,
            vec![Tag {
                at_s: 0.2,
                tag: "<spk:1>".into()
            }]
        );
        assert!((words[0].confidence - 1.0).abs() < 1e-12);
    }

    #[test]
    fn la_normalisation_ignore_le_remplissage() {
        let mut mel = vec![vec![1.0, 3.0, 9.0]];
        normalize_per_feature(&mut mel, 2);
        assert!((mel[0][0] + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4);
        assert!((mel[0][1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4);
        assert_eq!(mel[0][2], 0.0);
    }

    #[test]
    fn un_audio_vide_ne_fait_aucune_fenetre() {
        assert!(plan_windows(0, 3000, 8, 4).is_empty());
    }

    #[test]
    fn la_parole_est_le_complement_du_blanc() {
        let p = speech_probability(&[0.0, f32::NEG_INFINITY]);
        assert_eq!(p, vec![0.0, 1.0]);
    }
}
