"""Écrire les fixtures de parité de l'hôte ASR à partir du jumeau Python.

- ``tests/fixtures/asr_meta.onnx`` : le vrai graphe réduit à ses métadonnées
  (banc de filtres, fenêtre, vocabulaire), sans un octet de poids ;
- ``tests/fixtures/asr_reference.json`` : 1,5 s de parole réelle et son log-mel
  normalisé par ``vocal_helper.nemotron_asr`` ; les sorties du vrai graphe sur
  30 s, puis les mots et balises que l'hôte Python en tire.

Usage (avec le venv de vocal-helper) :

    python scripts/make_asr_fixture.py --graph <nemotron-asr-ctc.int8.onnx> --audio <mix.wav>
"""

from __future__ import annotations

import json
import pathlib

import click
import numpy as np
import soundfile as sf

from vocal_helper.nemotron import _graph_metadata
from vocal_helper.nemotron_asr import (
    NemotronOnnxAsr,
    greedy_ctc,
    log_mel,
    normalize_per_feature,
    tokens_to_words,
)

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests/fixtures"


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


@click.command()
@click.option("--graph", type=click.Path(path_type=pathlib.Path, exists=True), required=True)
@click.option("--audio", type=click.Path(path_type=pathlib.Path, exists=True), required=True)
@click.option("--at", default=120.0, show_default=True, help="Début de l'extrait, en secondes.")
def main(graph: pathlib.Path, audio: pathlib.Path, at: float) -> None:
    """Écrire les deux fixtures."""
    meta = _graph_metadata(graph)
    body = b"".join(
        _field(14, _field(1, k.encode()) + _field(2, v.encode())) for k, v in sorted(meta.items())
    )
    (FIXTURES / "asr_meta.onnx").write_bytes(body)

    signal, sr = sf.read(audio, dtype="float32")
    assert sr == 16000
    start = int(at * sr)
    short = signal[start : start + 24000]
    asr = NemotronOnnxAsr(graph, providers=["CPUExecutionProvider"])
    s = asr.settings
    mel, valid = log_mel(short, s)
    mel = normalize_per_feature(mel, valid)

    clip = signal[start : start + 30 * sr]
    ids, best, blank = asr.frames(clip, sr)
    words, tags = tokens_to_words(greedy_ctc(ids, best, s.blank_id), s.vocab, s.frame_s)
    reference = {
        "mel": {
            "audio": [round(float(v), 7) for v in short],
            "valid": valid,
            "frames": [[round(float(v), 5) for v in row] for row in mel],
        },
        "ctc": {
            "ids": ids.astype(int).tolist(),
            "logprob": [round(float(v), 6) for v in best],
            "words": [
                {"text": w.text, "start_s": w.start_s, "end_s": w.end_s, "confidence": w.confidence}
                for w in words
            ],
            "tags": [{"at_s": t, "tag": tag} for t, tag in tags],
        },
    }
    (FIXTURES / "asr_reference.json").write_text(json.dumps(reference, ensure_ascii=False))
    print(f"{len(meta)} métadonnées, {valid} trames mel, {len(ids)} trames CTC, {len(words)} mots")


if __name__ == "__main__":
    main()
