//! Ouvrir une session ONNX Runtime sur le bon moteur (feature `onnx`).
//!
//! Le pont entre la politique pure de [`crate::accel`] et la caisse `ort` :
//! enregistrer les moteurs dans l'ordre, retomber sur le CPU quand un moteur
//! refuse, mesurer quand personne n'a rien demandé, garder le verdict.
//!
//! Tout ce qui peut manquer à l'exécution — pilote NVIDIA, CUDA 13, cuDNN 9,
//! bibliothèques TensorRT — se traduit par un avertissement et un repli, jamais
//! par un échec d'ouverture : seul un graphe illisible fait échouer.

use std::fmt::Display;
use std::path::Path;
use std::time::{Duration, Instant};

use ort::environment::Environment;
#[allow(unused_imports)]
use ort::ep::ExecutionProvider;
use ort::session::builder::SessionBuilder;
use ort::session::Session;

use crate::accel::{
    choose, graph_digest, load_verdict, measure, register_in_order, requested_order, save_verdict,
    verdict_dir, AccelError, Provider, Registrar, Registration, Verdict, PROBE_AUDIO, PROBE_WARMUP,
};

/// Une session ouverte, avec les moteurs qu'elle a vraiment retenus.
#[derive(Debug)]
pub struct Opened {
    /// La session.
    pub session: Session,
    /// Les moteurs enregistrés, dans l'ordre ; le CPU en dernier.
    pub providers: Vec<Provider>,
    /// Ce qui a été sauté en chemin, déjà écrit sur la sortie d'erreur.
    pub warnings: Vec<String>,
}

struct BuilderRegistrar<'a>(&'a mut SessionBuilder);

impl Registrar for BuilderRegistrar<'_> {
    #[allow(unused_variables)]
    fn register(&mut self, provider: Provider) -> Result<(), String> {
        let builder = &mut *self.0;
        let result: ort::Result<()> = match provider {
            #[cfg(feature = "cuda")]
            Provider::Cuda => ort::ep::CUDA::default().register(builder),
            #[cfg(feature = "tensorrt")]
            Provider::TensorRt => ort::ep::TensorRT::default().register(builder),
            #[cfg(feature = "coreml")]
            Provider::CoreMl => ort::ep::CoreML::default().register(builder),
            #[cfg(feature = "directml")]
            Provider::DirectMl => ort::ep::DirectML::default().register(builder),
            #[cfg(feature = "openvino")]
            Provider::OpenVino => ort::ep::OpenVINO::default().register(builder),
            Provider::Cpu => Ok(()),
            #[allow(unreachable_patterns)]
            _ => return Err("non compilé dans cette caisse".into()),
        };
        result.map_err(|e| e.to_string())
    }
}

fn warn(warnings: &[String]) {
    for w in warnings {
        eprintln!("vocal-helper-rs : {w}");
    }
}

/// Ouvrir `path` en enregistrant `order` dans l'ordre, le CPU en dernier.
///
/// Un moteur qui refuse de s'enregistrer est sauté ; si la session refuse
/// ensuite de se construire avec les moteurs retenus (TensorRT qui ne sait pas
/// compiler le graphe, mémoire GPU insuffisante), elle est reconstruite sur le
/// CPU seul. Chaque repli est écrit sur la sortie d'erreur.
///
/// # Erreurs
///
/// [`ort::Error`] seulement si le CPU lui-même ne peut pas ouvrir le graphe.
pub fn open_session(
    environment: &Environment,
    path: &Path,
    order: &[Provider],
) -> ort::Result<Opened> {
    let mut builder = Session::builder(environment)?;
    let Registration {
        active,
        mut warnings,
    } = register_in_order(order, &mut BuilderRegistrar(&mut builder));
    let session = match builder.commit_from_file(path) {
        Ok(session) => session,
        Err(e) if active.len() > 1 => {
            warnings.push(format!(
                "la session ne se construit pas sur {} ({e}) ; repli sur le CPU seul",
                active[0].as_str()
            ));
            warn(&warnings);
            let session = Session::builder(environment)?.commit_from_file(path)?;
            return Ok(Opened {
                session,
                providers: vec![Provider::Cpu],
                warnings,
            });
        }
        Err(e) => return Err(e),
    };
    warn(&warnings);
    Ok(Opened {
        session,
        providers: active,
        warnings,
    })
}

/// De l'audio synthétique à 16 kHz pour la mesure : le calcul d'un graphe à
/// formes fixes ne dépend pas de ce qu'on lui dit.
fn probe_audio(duration: Duration) -> Vec<f32> {
    let n = (duration.as_secs_f64() * 16_000.0) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.05 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.02 * (2.0 * std::f32::consts::PI * 3_100.0 * t).sin()
        })
        .collect()
}

/// Ouvrir un graphe sur le moteur décidé par l'ordre d'autorité de
/// [`crate::accel`], et rendre ce que `wrap` construit autour de la session.
///
/// `probe` fait tourner le graphe sur un tampon à 16 kHz ; il sert à la mesure.
pub(crate) fn open_accelerated<S, E>(
    environment: &Environment,
    path: &Path,
    requested: Option<&[Provider]>,
    mut wrap: impl FnMut(Session) -> S,
    mut probe: impl FnMut(&mut S, &[f32]) -> Result<(), E>,
) -> Result<(S, Vec<Provider>), E>
where
    E: From<ort::Error> + From<AccelError> + Display,
{
    if let Some(order) = requested_order(requested)? {
        let opened = open_session(environment, path, &order)?;
        return Ok((wrap(opened.session), opened.providers));
    }

    let candidates = Provider::compiled();
    if candidates == [Provider::Cpu] {
        let opened = open_session(environment, path, &[Provider::Cpu])?;
        return Ok((wrap(opened.session), opened.providers));
    }

    let digest = graph_digest(path).ok();
    let dir = verdict_dir();
    if let (Some(digest), Some(dir)) = (&digest, &dir) {
        if let Some(verdict) =
            load_verdict(dir, digest).filter(|v| v.still_holds(digest, &candidates))
        {
            let opened = open_session(environment, path, &[verdict.chosen])?;
            return Ok((wrap(opened.session), opened.providers));
        }
    }

    let warmup = probe_audio(PROBE_WARMUP);
    let timed = probe_audio(PROBE_AUDIO);
    let (timings, best, warnings) = measure(
        &candidates,
        PROBE_AUDIO.as_secs_f64(),
        |provider| -> Result<Option<(S, Vec<Provider>)>, E> {
            let opened = open_session(environment, path, &[provider])?;
            if opened.providers[0] != provider {
                return Ok(None);
            }
            Ok(Some((wrap(opened.session), opened.providers)))
        },
        |(session, _)| {
            probe(session, &warmup)?;
            let start = Instant::now();
            probe(session, &timed)?;
            Ok(start.elapsed())
        },
    );
    warn(&warnings);

    if let (Some(chosen), Some(digest), Some(dir)) = (choose(&timings), digest, dir) {
        let verdict = Verdict {
            chosen,
            timings,
            graph_digest: digest,
            tried: candidates,
        };
        if let Err(e) = save_verdict(&dir, &verdict) {
            warn(&[format!(
                "verdict de mesure non gardé dans {} ({e}) ; il sera remesuré",
                dir.display()
            )]);
        }
    }

    match best {
        Some((_, (session, providers))) => Ok((session, providers)),
        None => {
            let opened = open_session(environment, path, &[Provider::Cpu])?;
            Ok((wrap(opened.session), opened.providers))
        }
    }
}
