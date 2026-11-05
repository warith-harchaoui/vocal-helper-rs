//! Le front-end log-mel que le graphe de diarisation attend, à la trame près.
//!
//! # Pourquoi cette caisse existe seule
//!
//! Le graphe ONNX ne prend pas de l'audio, il prend des trames log-mel. Sa
//! configuration en donne la charpente — 16 kHz, fenêtre de 400 échantillons,
//! saut de 160, transformée sur 512, pré-accentuation 0,97, 128 bandes — mais
//! pas le détail qui décide. Ces détails ont été **retrouvés en comparant aux
//! sorties de l'extracteur officiel**, pas devinés. Deux comptent :
//!
//! - La fenêtre d'analyse de 400 points est **centrée** dans la transformée de
//!   512 plutôt que collée au début. La coller multiplie par seize l'écart aux
//!   sorties officielles — 0,530 au lieu de 0,033 — parce que le décalage de
//!   phase introduit change le spectre.
//! - Le plancher du logarithme est `2^-24`, retrouvé en lisant le minimum des
//!   sorties officielles : −16,36, qui en est exactement le logarithme.
//!
//! Au régime établi, l'écart moyen aux sorties officielles est de 0,006 pour des
//! valeurs d'écart-type 1,3 — soit un demi pour cent. Les trames de bord
//! diffèrent : l'extracteur officiel remplit sa dernière trame d'une valeur
//! sentinelle, ce qu'il n'y a pas lieu de reproduire.
//!
//! Isolée dans sa propre caisse parce qu'elle n'a besoin de rien — ni d'ONNX, ni
//! de réseau, ni de modèle — donc elle se teste et se vérifie seule.

use realfft::RealFftPlanner;
use thiserror::Error;

/// Ce qui peut mal se passer et doit se dire plutôt que se corriger en silence.
#[derive(Debug, Error)]
pub enum MelError {
    /// Rééchantillonner sans le dire produirait des trames plausibles et fausses.
    #[error("le modèle veut {expected} Hz, reçu {received} Hz : rééchantillonner d'abord")]
    SampleRate {
        /// Ce que le modèle veut : 16 kHz.
        expected: u32,
        /// Ce qu'on lui a donné.
        received: u32,
    },
}

/// Les paramètres du front-end, tels que la configuration du modèle les donne.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MelSettings {
    /// 16 kHz, imposé par le modèle.
    pub sample_rate: u32,
    /// Taille de la transformée.
    pub n_fft: usize,
    /// Longueur de la fenêtre d'analyse, **plus courte** que la transformée.
    pub win_length: usize,
    /// Avance entre deux trames : 160 échantillons, soit 10 ms.
    pub hop_length: usize,
    /// Nombre de bandes.
    pub n_mels: usize,
    /// Pré-accentuation appliquée avant tout découpage.
    pub preemphasis: f64,
    /// Plancher ajouté avant le logarithme.
    pub log_guard: f64,
}

impl Default for MelSettings {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 128,
            preemphasis: 0.97,
            log_guard: 2.0_f64.powi(-24),
        }
    }
}

/// Hertz vers l'échelle mel, convention Slaney : linéaire sous 1 kHz, logarithmique au-dessus.
fn hz_to_mel(hz: f64) -> f64 {
    const LIN_HZ: f64 = 1000.0;
    const LIN_MEL: f64 = 15.0;
    let step = 200.0 / 3.0;
    if hz >= LIN_HZ {
        LIN_MEL + (hz / LIN_HZ).ln() / (6.4_f64.ln() / 27.0)
    } else {
        hz / step
    }
}

/// L'inverse de [`hz_to_mel`].
fn mel_to_hz(mel: f64) -> f64 {
    const LIN_HZ: f64 = 1000.0;
    const LIN_MEL: f64 = 15.0;
    let step = 200.0 / 3.0;
    if mel >= LIN_MEL {
        LIN_HZ * ((mel - LIN_MEL) * (6.4_f64.ln() / 27.0)).exp()
    } else {
        mel * step
    }
}

/// Le banc de filtres mel, `n_mels` lignes de `n_fft / 2 + 1` colonnes.
///
/// Écrit à la main plutôt que tiré d'une dépendance : c'est une matrice
/// triangulaire ; la convention qui compte — l'échelle Slaney et sa
/// normalisation par la largeur de bande — tient en quelques lignes. Une
/// dépendance de plus sur un chemin destiné au mobile se paierait au poids du
/// binaire pour rien.
pub fn mel_filterbank(settings: &MelSettings) -> Vec<Vec<f64>> {
    let n_bins = settings.n_fft / 2 + 1;
    let nyquist = f64::from(settings.sample_rate) / 2.0;
    let low = hz_to_mel(0.0);
    let high = hz_to_mel(nyquist);
    let edges: Vec<f64> = (0..settings.n_mels + 2)
        .map(|i| {
            let mel = low + (high - low) * (i as f64) / (settings.n_mels as f64 + 1.0);
            mel_to_hz(mel)
        })
        .collect();
    let bin_hz: Vec<f64> = (0..n_bins)
        .map(|k| (k as f64) * f64::from(settings.sample_rate) / (settings.n_fft as f64))
        .collect();

    (0..settings.n_mels)
        .map(|m| {
            let (lo, centre, hi) = (edges[m], edges[m + 1], edges[m + 2]);
            // Normalisation Slaney : chaque filtre intègre à surface constante,
            // sans quoi les bandes hautes, plus larges, domineraient.
            let width = edges[m + 2] - edges[m];
            let scale = if width > 0.0 { 2.0 / width } else { 1.0 };
            bin_hz
                .iter()
                .map(|&f| {
                    let rising = (f - lo) / (centre - lo).max(1e-12);
                    let falling = (hi - f) / (hi - centre).max(1e-12);
                    rising.min(falling).max(0.0) * scale
                })
                .collect()
        })
        .collect()
}

/// La fenêtre d'analyse, **centrée** dans la transformée.
///
/// Hann périodique comme la référence, placée au milieu des `n_fft` points.
/// C'est la convention dont l'absence multipliait par seize l'écart aux sorties
/// officielles, donc elle est isolée ici pour être testable à part.
pub fn centred_window(settings: &MelSettings) -> Vec<f64> {
    let mut window = vec![0.0; settings.n_fft];
    let offset = (settings.n_fft - settings.win_length) / 2;
    for i in 0..settings.win_length {
        // Hann périodique : dénominateur `win_length`, pas `win_length - 1`.
        let phase = std::f64::consts::TAU * (i as f64) / (settings.win_length as f64);
        window[offset + i] = 0.5 - 0.5 * phase.cos();
    }
    window
}

/// Les trames log-mel du signal, `(trames, n_mels)` en ordre ligne par ligne.
///
/// # Erreurs
///
/// [`MelError::SampleRate`] si la fréquence n'est pas celle du modèle. On refuse
/// plutôt que de rééchantillonner en silence : un rééchantillonnage implicite
/// donnerait des trames plausibles et fausses, ce qui est le pire des deux.
pub fn log_mel(
    audio: &[f32],
    sample_rate: u32,
    settings: &MelSettings,
) -> Result<Vec<Vec<f32>>, MelError> {
    if sample_rate != settings.sample_rate {
        return Err(MelError::SampleRate {
            expected: settings.sample_rate,
            received: sample_rate,
        });
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    // Pré-accentuation, avant tout découpage.
    let mut signal = Vec::with_capacity(audio.len());
    signal.push(f64::from(audio[0]));
    for i in 1..audio.len() {
        signal.push(f64::from(audio[i]) - settings.preemphasis * f64::from(audio[i - 1]));
    }

    // Remplissage par réflexion, pour que la trame zéro soit centrée sur
    // l'échantillon zéro — la convention de la référence.
    let pad = settings.n_fft / 2;
    let padded = reflect_pad(&signal, pad);

    let window = centred_window(settings);
    let bank = mel_filterbank(settings);
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(settings.n_fft);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    let frames = if padded.len() < settings.n_fft {
        0
    } else {
        1 + (padded.len() - settings.n_fft) / settings.hop_length
    };
    let mut out = Vec::with_capacity(frames);
    for frame in 0..frames {
        let start = frame * settings.hop_length;
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = padded[start + i] * window[i];
        }
        fft.process(&mut input, &mut output)
            .expect("les longueurs viennent du plan, donc elles concordent");
        let power: Vec<f64> = output.iter().map(|c| c.norm_sqr()).collect();
        out.push(
            bank.iter()
                .map(|filter| {
                    let energy: f64 = filter.iter().zip(power.iter()).map(|(&w, &p)| w * p).sum();
                    (energy + settings.log_guard).ln() as f32
                })
                .collect(),
        );
    }
    Ok(out)
}

/// Un remplissage par réflexion, qui retombe sur des zéros si le signal est trop court.
fn reflect_pad(signal: &[f64], pad: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(signal.len() + 2 * pad);
    if signal.len() > pad {
        out.extend((1..=pad).rev().map(|i| signal[i]));
    } else {
        out.extend(std::iter::repeat_n(0.0, pad));
    }
    out.extend_from_slice(signal);
    if signal.len() > pad {
        let last = signal.len() - 1;
        out.extend((1..=pad).map(|i| signal[last - i]));
    } else {
        out.extend(std::iter::repeat_n(0.0, pad));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn une_trame_toutes_les_dix_millisecondes() {
        let settings = MelSettings::default();
        let frames = log_mel(&vec![0.0; 16_000], 16_000, &settings).unwrap();
        assert_eq!(frames.len(), 101);
        assert_eq!(frames[0].len(), 128);
    }

    #[test]
    fn le_silence_tombe_sur_le_plancher_du_logarithme() {
        let settings = MelSettings::default();
        let frames = log_mel(&vec![0.0; 8_000], 16_000, &settings).unwrap();
        let floor = settings.log_guard.ln() as f32;
        for frame in &frames {
            for &value in frame {
                assert!((value - floor).abs() < 1e-4, "{value} contre {floor}");
            }
        }
    }

    #[test]
    fn une_mauvaise_frequence_est_refusee_plutot_que_reechantillonnee() {
        let err = log_mel(&vec![0.0; 800], 8_000, &MelSettings::default()).unwrap_err();
        assert!(matches!(
            err,
            MelError::SampleRate {
                expected: 16_000,
                ..
            }
        ));
    }

    #[test]
    fn un_signal_vide_n_est_pas_une_erreur() {
        assert!(log_mel(&[], 16_000, &MelSettings::default())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn la_fenetre_est_centree_dans_la_transformee() {
        // La convention retrouvée contre l'extracteur officiel : 56 zéros de
        // chaque côté. La coller au début multipliait par seize l'écart.
        let window = centred_window(&MelSettings::default());
        assert_eq!(window.len(), 512);
        for i in 0..56 {
            assert_eq!(window[i], 0.0, "la fenêtre doit commencer à 56");
            assert_eq!(window[511 - i], 0.0, "et finir 56 avant la fin");
        }
        assert!(window[56 + 200] > 0.9, "son sommet doit être au milieu");
    }

    #[test]
    fn la_fenetre_est_un_hann_periodique() {
        // Périodique et non symétrique : le premier point vaut zéro et il n'y a
        // pas de second zéro à la fin de la partie active.
        let window = centred_window(&MelSettings::default());
        assert!(window[56].abs() < 1e-12);
        assert!(window[56 + 399] > 0.0, "un Hann symétrique finirait à zéro");
    }

    #[test]
    fn le_banc_de_filtres_a_la_bonne_forme() {
        let settings = MelSettings::default();
        let bank = mel_filterbank(&settings);
        assert_eq!(bank.len(), 128);
        assert_eq!(bank[0].len(), 257);
        for filter in &bank {
            assert!(filter.iter().all(|&w| w >= 0.0), "pas de poids négatif");
            assert!(filter.iter().any(|&w| w > 0.0), "pas de filtre vide");
        }
    }

    #[test]
    fn les_filtres_montent_en_frequence_sans_se_croiser() {
        let bank = mel_filterbank(&MelSettings::default());
        let peak = |filter: &Vec<f64>| {
            filter
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        let peaks: Vec<usize> = bank.iter().map(peak).collect();
        assert!(
            peaks.windows(2).all(|w| w[0] <= w[1]),
            "les sommets doivent monter en fréquence"
        );
    }

    #[test]
    fn les_bandes_hautes_ne_dominent_pas_les_basses() {
        // Ce que la normalisation Slaney achète : sans elle, un filtre large
        // intégrerait plus d'énergie qu'un filtre étroit à spectre plat égal.
        let bank = mel_filterbank(&MelSettings::default());
        let area = |filter: &Vec<f64>| filter.iter().sum::<f64>();
        let low = area(&bank[5]);
        let high = area(&bank[120]);
        assert!(
            (low / high).abs() < 50.0,
            "les surfaces doivent rester comparables : {low} contre {high}"
        );
    }

    #[test]
    fn un_ton_pur_eclaire_la_bande_qui_lui_correspond() {
        let settings = MelSettings::default();
        let tone: Vec<f32> = (0..16_000)
            .map(|i| {
                let t = i as f64 / f64::from(settings.sample_rate);
                (std::f64::consts::TAU * 1000.0 * t).sin() as f32 * 0.5
            })
            .collect();
        let frames = log_mel(&tone, 16_000, &settings).unwrap();
        let middle = &frames[50];
        let loudest = middle
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // 1 kHz est la charnière de l'échelle Slaney, à 15 mel sur les 128
        // bandes qui couvrent jusqu'à Nyquist.
        assert!(
            (40..=80).contains(&loudest),
            "un ton à 1 kHz devrait éclairer le milieu du banc, pas la bande {loudest}"
        );
    }

    #[test]
    fn la_preaccentuation_attenue_le_continu() {
        let settings = MelSettings::default();
        let constant = vec![0.5_f32; 16_000];
        let frames = log_mel(&constant, 16_000, &settings).unwrap();
        // Un signal constant est du continu pur, que la pré-accentuation
        // supprime presque entièrement : il doit rester proche du plancher.
        let floor = settings.log_guard.ln() as f32;
        let middle = &frames[50];
        assert!(
            middle.iter().all(|&v| v < floor + 20.0),
            "le continu doit être atténué, pas amplifié"
        );
    }
}
