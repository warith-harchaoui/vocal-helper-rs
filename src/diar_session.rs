//! Piloter le graphe de diarisation Nemotron 3 par ONNX Runtime (feature `onnx`).
//!
//! Le portage de `vocal_helper.nemotron.NemotronOnnxDiar.probabilities` : le
//! fichier défile comme un flux, morceau par morceau ; le cache de locuteurs
//! et la file sont tenus par [`crate::cache::streaming_update`] entre deux pas.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use thiserror::Error;

use crate::accel::{AccelError, Provider};
use crate::cache::{streaming_update, StreamingState};
use crate::mel::{log_mel, MelError, MelSettings};
use crate::onnx_meta::{base64_f32le, graph_metadata, MetaError};
use crate::schedule::{Schedule, META_PREFIX};
use crate::segments::{segments_from_probabilities, Segment};

/// Ce qui empêche de diariser.
#[derive(Debug, Error)]
pub enum DiarError {
    /// Métadonnées illisibles.
    #[error(transparent)]
    Meta(#[from] MetaError),
    /// Mauvaise fréquence.
    #[error(transparent)]
    Mel(#[from] MelError),
    /// Le graphe ne porte pas ce que ce point de contrôle demande.
    #[error("{0}")]
    Graph(String),
    /// ONNX Runtime a échoué.
    #[error("onnxruntime : {0}")]
    Ort(#[from] ort::Error),
    /// `VOCAL_HELPER_PROVIDER` nomme un moteur inconnu.
    #[error(transparent)]
    Accel(#[from] AccelError),
}

/// Une session ouverte sur le graphe de diarisation.
pub struct NemotronDiar {
    schedule: Schedule,
    silence: Vec<f32>,
    session: Session,
    providers: Vec<Provider>,
    _environment: ort::environment::Environment,
}

impl NemotronDiar {
    /// Ouvrir le graphe, avec le plus long calendrier que `latency_s` permet.
    ///
    /// Le calendrier des métadonnées est celui hors ligne, qui dépasse la
    /// séquence que le graphe accepte : il est **toujours** remplacé, comme dans
    /// le jumeau Python.
    ///
    /// Le moteur d'exécution suit l'ordre d'autorité de [`crate::accel`] ; voir
    /// [`Self::open_with`].
    ///
    /// # Erreurs
    ///
    /// [`DiarError`] si le graphe est illisible, sans empreinte de silence, voire
    /// si ONNX Runtime refuse de l'ouvrir même sur le CPU.
    pub fn open(path: &Path, latency_s: Option<f64>) -> Result<Self, DiarError> {
        Self::open_with(path, latency_s, None)
    }

    /// Ouvrir le graphe sur les moteurs `providers`, dans cet ordre.
    ///
    /// `None` (ou une liste vide) laisse décider `VOCAL_HELPER_PROVIDER` puis la
    /// mesure sur l'appareil. Un moteur qui ne s'enregistre pas est sauté avec
    /// un avertissement : la session s'ouvre toujours, au pire sur le CPU.
    ///
    /// # Erreurs
    ///
    /// [`DiarError`] si le graphe est illisible, sans empreinte de silence, si
    /// `VOCAL_HELPER_PROVIDER` nomme un moteur inconnu, voire si ONNX Runtime
    /// refuse de l'ouvrir même sur le CPU.
    pub fn open_with(
        path: &Path,
        latency_s: Option<f64>,
        providers: Option<&[Provider]>,
    ) -> Result<Self, DiarError> {
        let meta = graph_metadata(path)?;
        let schedule = Schedule::from_metadata(&meta).within(latency_s);
        if !schedule.fits() {
            return Err(DiarError::Graph(format!(
                "calendrier de {}+{} trames impossible pour ce graphe",
                schedule.chunk_len, schedule.chunk_right_context
            )));
        }
        let key = format!("{META_PREFIX}learnable_sil_emb_f32le_b64");
        let silence = base64_f32le(meta.get(&key).ok_or_else(|| {
            DiarError::Graph(format!(
                "{} ne porte pas l'empreinte de silence",
                path.display()
            ))
        })?)?;
        if silence.len() != schedule.emb_dim {
            return Err(DiarError::Graph(format!(
                "empreinte de silence de {} valeurs pour {}",
                silence.len(),
                schedule.emb_dim
            )));
        }
        let environment = ort::init().with_name("vocal-helper-rs").build()?;
        let (mut diar, active) = crate::runtime::open_accelerated(
            &environment,
            path,
            providers,
            |session| Self {
                schedule,
                silence: silence.clone(),
                session,
                providers: Vec::new(),
                _environment: environment.clone(),
            },
            |diar: &mut Self, audio: &[f32]| diar.probabilities(audio, 16_000).map(|_| ()),
        )?;
        diar.providers = active;
        Ok(diar)
    }

    /// Les moteurs que la session a retenus, dans l'ordre ; le CPU en dernier.
    pub fn providers(&self) -> &[Provider] {
        &self.providers
    }

    /// Le calendrier effectif.
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// Les probabilités d'activité par locuteur à 10 ms, ligne par ligne
    /// (`num_speakers` valeurs par trame).
    ///
    /// # Erreurs
    ///
    /// [`DiarError`] sur une mauvaise fréquence ou un échec d'inférence.
    pub fn probabilities(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>, DiarError> {
        let schedule = self.schedule;
        let settings = MelSettings {
            n_mels: schedule.n_mels,
            ..MelSettings::default()
        };
        let frames = log_mel(audio, sample_rate, &settings)?;
        let n_mels = schedule.n_mels;
        let n = schedule.num_speakers;
        let sub = schedule.subsampling;
        let step = schedule.chunk_len * sub;
        let right = schedule.chunk_right_context * sub;
        let mut state = StreamingState::new();
        let mut collected = Vec::new();

        let mut start = 0;
        while start < frames.len() {
            let end = (start + step + right).min(frames.len());
            let usable = ((end - start) / sub) * sub;
            if usable < sub {
                break;
            }
            let mut chunk = Vec::with_capacity(usable * n_mels);
            for row in &frames[start..start + usable] {
                chunk.extend_from_slice(row);
            }
            let emb = schedule.emb_dim;
            let inputs = ort::inputs![
                "chunk" => Tensor::from_array(([1usize, usable, n_mels], chunk))?,
                "chunk_lengths" => Tensor::from_array(([1usize], vec![usable as i64]))?,
                "spkcache" => Tensor::from_array(([1usize, state.spkcache.len(), emb], state.spkcache_flat(emb)))?,
                "spkcache_lengths" => Tensor::from_array(([1usize], vec![state.spkcache.len() as i64]))?,
                "fifo" => Tensor::from_array(([1usize, state.fifo.len(), emb], state.fifo_flat(emb)))?,
                "fifo_lengths" => Tensor::from_array(([1usize], vec![state.fifo.len() as i64]))?,
            ];
            let outputs = self.session.run(inputs)?;
            let (_, coarse) = outputs["spkcache_fifo_chunk_preds"].try_extract_tensor::<f32>()?;
            let (_, fine) = outputs["spkcache_fifo_chunk_preds_hr"].try_extract_tensor::<f32>()?;
            let (shape, embs) = outputs["chunk_pre_encode_embs"].try_extract_tensor::<f32>()?;
            let chunk_frames = shape[1] as usize;
            let coarse_frames = coarse.len() / n;

            // Le contexte droit a servi à mieux décider du morceau ; il n'en fait
            // pas partie, sauf au dernier morceau où il n'y a rien à retirer.
            let mut kept = (usable / sub).saturating_sub(schedule.chunk_right_context);
            if start + step >= frames.len() {
                kept = usable / sub;
            }
            let kept = kept.min(chunk_frames);
            let embeddings: Vec<Vec<f32>> = embs
                .chunks_exact(emb)
                .take(kept)
                .map(<[f32]>::to_vec)
                .collect();
            streaming_update(&mut state, &embeddings, coarse, &schedule, &self.silence);

            let head = (coarse_frames - chunk_frames) * schedule.upsample_factor;
            let span = kept * schedule.upsample_factor;
            collected.extend_from_slice(&fine[head * n..(head + span) * n]);
            start += step;
        }
        Ok(collected)
    }

    /// Les tours de parole, au seuil `threshold`, recollés sous 0,5 s.
    ///
    /// # Erreurs
    ///
    /// [`DiarError`] sur une mauvaise fréquence ou un échec d'inférence.
    pub fn diarize(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        threshold: f32,
    ) -> Result<Vec<Segment>, DiarError> {
        let probabilities = self.probabilities(audio, sample_rate)?;
        Ok(segments_from_probabilities(
            &probabilities,
            self.schedule.num_speakers,
            0.01,
            threshold,
            0.5,
        ))
    }
}
