//! Piloter le graphe `nemotron-asr-ctc` par ONNX Runtime (feature `onnx`).
//!
//! Le portage de `vocal_helper.nemotron_asr.NemotronOnnxAsr.transcribe` :
//! métadonnées lues sans charger les poids, log-mel normalisé sur tout le
//! fichier, fenêtres de 30 s, puis décodage par [`crate::asr`].

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;
use thiserror::Error;

use crate::asr::{
    greedy_ctc, log_mel, normalize_per_feature, plan_windows, speech_probability, tokens_to_words,
    AsrError, AsrSettings, Tag, Word,
};
use crate::mel::MelError;
use crate::onnx_meta::{graph_metadata, MetaError};

/// Ce qui empêche de transcrire.
#[derive(Debug, Error)]
pub enum SessionError {
    /// Métadonnées illisibles.
    #[error(transparent)]
    Meta(#[from] MetaError),
    /// Métadonnées hors contrat.
    #[error(transparent)]
    Asr(#[from] AsrError),
    /// Mauvaise fréquence.
    #[error(transparent)]
    Mel(#[from] MelError),
    /// ONNX Runtime a échoué.
    #[error("onnxruntime : {0}")]
    Ort(#[from] ort::Error),
}

/// Par trame de 40 ms : jeton, sa log-probabilité, celle du blanc.
pub type Frames = (Vec<i32>, Vec<f32>, Vec<f32>);

/// Le résultat d'une transcription.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    /// Les mots horodatés.
    pub words: Vec<Word>,
    /// Les balises : locuteur, événement, style.
    pub tags: Vec<Tag>,
    /// Probabilité de parole par trame de 40 ms.
    pub speech: Vec<f32>,
}

impl Transcript {
    /// Le texte, mots séparés par une espace.
    pub fn text(&self) -> String {
        self.words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Une session ouverte sur le graphe ASR.
pub struct NemotronAsr {
    settings: AsrSettings,
    session: Session,
    path: PathBuf,
    _environment: ort::environment::Environment,
}

impl NemotronAsr {
    /// Ouvrir le graphe ; les réglages viennent de ses métadonnées.
    ///
    /// # Erreurs
    ///
    /// [`SessionError`] si le graphe est illisible ou hors contrat, ou bien si ONNX
    /// Runtime refuse de l'ouvrir.
    pub fn open(path: &Path) -> Result<Self, SessionError> {
        let settings = AsrSettings::from_metadata(&graph_metadata(path)?)?;
        let environment = ort::init().with_name("vocal-helper-rs").build()?;
        let session = Session::builder(&environment)?.commit_from_file(path)?;
        Ok(Self {
            settings,
            session,
            path: path.to_owned(),
            _environment: environment,
        })
    }

    /// Les réglages lus dans le graphe.
    pub fn settings(&self) -> &AsrSettings {
        &self.settings
    }

    /// Le chemin du graphe.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Jetons, log-probabilités et log-probabilités du blanc, toutes les 40 ms.
    ///
    /// # Erreurs
    ///
    /// [`SessionError`] sur une mauvaise fréquence ou un échec d'inférence.
    pub fn frames(&mut self, audio: &[f32], sample_rate: u32) -> Result<Frames, SessionError> {
        let s = &self.settings;
        let (mut mel, valid) = log_mel(audio, sample_rate, s)?;
        normalize_per_feature(&mut mel, valid);
        let chunk = (s.chunk_s * f64::from(s.sample_rate) / s.hop_length as f64).round() as usize;
        let plan = plan_windows(valid, chunk, s.stack, s.mel_per_frame());
        let n_mels = s.n_mels;
        let (mut ids, mut best, mut blank) = (Vec::new(), Vec::new(), Vec::new());
        for w in plan {
            let mut data = vec![0.0f32; n_mels * w.padded];
            for (band, row) in mel.iter().enumerate() {
                data[band * w.padded..band * w.padded + w.len]
                    .copy_from_slice(&row[w.start..w.start + w.len]);
            }
            let mel_t = Tensor::from_array(([1usize, n_mels, w.padded], data))?;
            let len_t = Tensor::from_array(([1usize], vec![w.enc_len]))?;
            let outputs = self
                .session
                .run(ort::inputs!["mel" => mel_t, "enc_len" => len_t])?;
            let (_, a) = outputs["token_ids"].try_extract_tensor::<i32>()?;
            let (_, b) = outputs["token_logprob"].try_extract_tensor::<f32>()?;
            let (_, c) = outputs["blank_logprob"].try_extract_tensor::<f32>()?;
            let keep = w.keep.min(a.len());
            ids.extend_from_slice(&a[..keep]);
            best.extend_from_slice(&b[..keep]);
            blank.extend_from_slice(&c[..keep]);
        }
        Ok((ids, best, blank))
    }

    /// Les mots horodatés, les balises et la probabilité de parole.
    ///
    /// # Erreurs
    ///
    /// [`SessionError`] sur une mauvaise fréquence ou un échec d'inférence.
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Transcript, SessionError> {
        let (ids, best, blank) = self.frames(audio, sample_rate)?;
        let s = &self.settings;
        let tokens = greedy_ctc(&ids, &best, s.blank_id);
        let (words, tags) = tokens_to_words(&tokens, &s.vocab, s.frame_s);
        Ok(Transcript {
            words,
            tags,
            speech: speech_probability(&blank),
        })
    }
}
