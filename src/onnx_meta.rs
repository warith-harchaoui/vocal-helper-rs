//! Les métadonnées d'un graphe ONNX, lues sans charger ses poids ni dépendre d'`onnx`.
//!
//! Les deux graphes du projet — diarisation et transcription — portent dans
//! leurs `metadata_props` tout ce que l'hôte doit savoir : calendrier, banc de
//! filtres, vocabulaire. Les lire ne doit pas coûter l'ouverture d'une session
//! sur des centaines de mégaoctets.
//!
//! Le format est du protobuf simple. On parcourt les champs de `ModelProto` en
//! sautant tout ce qui n'est pas `metadata_props` (champ 14, des paires
//! `StringStringEntryProto` clé 1 / valeur 2) : les poids sont enjambés par
//! `seek` et jamais lus. C'est le portage exact de
//! `vocal_helper.nemotron._graph_metadata`.

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use thiserror::Error;

/// Le numéro de champ de `metadata_props` dans `ModelProto` (onnx.proto).
const METADATA_FIELD: u64 = 14;

/// Ce qui empêche de lire les métadonnées.
#[derive(Debug, Error)]
pub enum MetaError {
    /// Le fichier n'a pas pu être lu.
    #[error("lecture du graphe : {0}")]
    Io(#[from] io::Error),
    /// Le fichier n'est pas un protobuf ONNX.
    #[error("pas un graphe ONNX : {0}")]
    Format(String),
}

fn varint<R: Read>(stream: &mut R) -> Result<Option<u64>, MetaError> {
    let (mut value, mut shift) = (0u64, 0u32);
    let mut byte = [0u8; 1];
    loop {
        if stream.read(&mut byte)? == 0 {
            return if shift == 0 {
                Ok(None)
            } else {
                Err(MetaError::Format("entier tronqué".into()))
            };
        }
        if shift >= 64 {
            return Err(MetaError::Format("entier trop long".into()));
        }
        value |= u64::from(byte[0] & 0x7F) << shift;
        shift += 7;
        if byte[0] & 0x80 == 0 {
            return Ok(Some(value));
        }
    }
}

fn entry(bytes: &[u8]) -> Result<(String, String), MetaError> {
    let mut cursor = io::Cursor::new(bytes);
    let (mut key, mut value) = (String::new(), String::new());
    while let Some(tag) = varint(&mut cursor)? {
        let length = varint(&mut cursor)?.unwrap_or(0) as usize;
        let start = cursor.position() as usize;
        let end = start
            .checked_add(length)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| MetaError::Format("paire tronquée".into()))?;
        let text = std::str::from_utf8(&bytes[start..end])
            .map_err(|_| MetaError::Format("texte non UTF-8".into()))?
            .to_owned();
        cursor.set_position(end as u64);
        match tag >> 3 {
            1 => key = text,
            2 => value = text,
            _ => {}
        }
    }
    Ok((key, value))
}

/// Lire les métadonnées d'un flux protobuf `ModelProto`.
///
/// # Erreurs
///
/// [`MetaError::Format`] si le flux n'est pas du protobuf lisible.
pub fn read_metadata<R: Read + Seek>(stream: &mut R) -> Result<HashMap<String, String>, MetaError> {
    let mut out = HashMap::new();
    while let Some(tag) = varint(stream)? {
        let (field, wire) = (tag >> 3, tag & 7);
        match wire {
            0 => {
                varint(stream)?;
            }
            1 => {
                stream.seek(SeekFrom::Current(8))?;
            }
            5 => {
                stream.seek(SeekFrom::Current(4))?;
            }
            2 => {
                let size = varint(stream)?
                    .ok_or_else(|| MetaError::Format(format!("champ {field} tronqué")))?;
                if field != METADATA_FIELD {
                    let size = i64::try_from(size)
                        .map_err(|_| MetaError::Format("taille démesurée".into()))?;
                    stream.seek(SeekFrom::Current(size))?;
                    continue;
                }
                let mut bytes = vec![0u8; size as usize];
                stream.read_exact(&mut bytes)?;
                let (key, value) = entry(&bytes)?;
                out.insert(key, value);
            }
            other => return Err(MetaError::Format(format!("type filaire {other}"))),
        }
    }
    Ok(out)
}

/// Lire les métadonnées d'un fichier `.onnx`.
///
/// # Erreurs
///
/// [`MetaError`] si le fichier est illisible ou n'est pas un graphe.
pub fn graph_metadata(path: &Path) -> Result<HashMap<String, String>, MetaError> {
    let mut file = io::BufReader::new(std::fs::File::open(path)?);
    read_metadata(&mut file)
}

/// Décoder du base64 standard (avec ou sans `=`), sans dépendance.
///
/// # Erreurs
///
/// [`MetaError::Format`] sur un caractère hors alphabet.
///
/// ```
/// assert_eq!(vocal_helper_rs::onnx_meta::base64_decode("aGk=").unwrap(), b"hi");
/// ```
pub fn base64_decode(text: &str) -> Result<Vec<u8>, MetaError> {
    fn sextet(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some(u32::from(c - b'A')),
            b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
            b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in text.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = sextet(c).ok_or_else(|| MetaError::Format("base64 invalide".into()))?;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// Des `f32` petit-boutistes encodés en base64.
///
/// # Erreurs
///
/// [`MetaError::Format`] si le base64 est invalide ou n'a pas un multiple de 4 octets.
pub fn base64_f32le(text: &str) -> Result<Vec<f32>, MetaError> {
    let bytes = base64_decode(text)?;
    if bytes.len() % 4 != 0 {
        return Err(MetaError::Format("flottants tronqués".into()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let byte = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    fn put_bytes(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
        put_varint(out, field << 3 | 2);
        put_varint(out, bytes.len() as u64);
        out.extend_from_slice(bytes);
    }

    fn graph(pairs: &[(&str, &str)]) -> Vec<u8> {
        let mut body = Vec::new();
        put_varint(&mut body, 1 << 3);
        put_varint(&mut body, 8);
        put_bytes(&mut body, 7, &[0u8; 1024]);
        for (k, v) in pairs {
            let mut e = Vec::new();
            put_bytes(&mut e, 1, k.as_bytes());
            put_bytes(&mut e, 2, v.as_bytes());
            put_bytes(&mut body, 14, &e);
        }
        body
    }

    #[test]
    fn les_poids_sont_enjambes_et_les_paires_lues() {
        let bytes = graph(&[("a.b", "1"), ("vocab", "[\"▁bon\"]")]);
        let meta = read_metadata(&mut io::Cursor::new(bytes)).unwrap();
        assert_eq!(meta.len(), 2);
        assert_eq!(meta["a.b"], "1");
        assert_eq!(meta["vocab"], "[\"▁bon\"]");
    }

    #[test]
    fn un_fichier_qui_n_est_pas_un_graphe_est_refuse() {
        let err = read_metadata(&mut io::Cursor::new(vec![0x0F, 0, 0])).unwrap_err();
        assert!(matches!(err, MetaError::Format(_)));
    }

    #[test]
    fn le_base64_aller_retour_des_flottants() {
        // [1.0, -2.5] en f32 petit-boutiste.
        let floats = base64_f32le("AACAPwAAIMA=").unwrap();
        assert_eq!(floats, vec![1.0, -2.5]);
        assert!(base64_decode("@@").is_err());
    }
}
