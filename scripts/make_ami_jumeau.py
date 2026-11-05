"""Écrire ce que le jumeau Python transcrit sur le sous-corpus AMI de vocal-helper.

``tests/fixtures/ami_jumeau_int8.json`` garde, pour chaque minute d'AMI, le texte
que ``vocal_helper.nemotron_asr`` tire du graphe int8 sur le CPU, avec la
version d'onnxruntime qui l'a produit. ``tests/jumeau_ami.rs`` transcrit les
mêmes minutes depuis Rust et mesure l'écart. Seul le texte est gardé : l'audio
reste dans le cache de vocal-helper (``~/.cache/vocal-helper/ami-subset``).

Usage (avec le venv de vocal-helper) :

    python scripts/make_ami_jumeau.py --graph <nemotron-asr-ctc.int8.onnx> --ami <ami-subset>
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time

import click
import onnxruntime
import soundfile as sf
from vocal_helper.nemotron_asr import NemotronOnnxAsr

FIXTURE = (
    pathlib.Path(__file__).resolve().parent.parent
    / "tests/fixtures/ami_jumeau_int8.json"
)


@click.command()
@click.option(
    "--graph", type=click.Path(exists=True, path_type=pathlib.Path), required=True
)
@click.option(
    "--ami", type=click.Path(exists=True, path_type=pathlib.Path), required=True
)
def main(graph: pathlib.Path, ami: pathlib.Path) -> None:
    """Transcrire chaque ``<clip>/clip.wav`` du sous-corpus et écrire la fixture."""
    asr = NemotronOnnxAsr(graph, providers=["CPUExecutionProvider"])
    clips, audio_s, compute_s = {}, 0.0, 0.0
    for wav in sorted(ami.glob("*/clip.wav")):
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        t0 = time.perf_counter()
        text = asr.transcribe(audio, sr).text
        compute_s += time.perf_counter() - t0
        audio_s += len(audio) / sr
        digest = hashlib.sha256(wav.read_bytes()).hexdigest()
        clips[wav.parent.name] = {"sha256": digest, "text": text}
    FIXTURE.write_text(
        json.dumps(
            {
                "graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
                "onnxruntime": onnxruntime.__version__,
                "clips": clips,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    )
    click.echo(f"{len(clips)} minutes, RTF {compute_s / audio_s:.3f} → {FIXTURE}")


if __name__ == "__main__":
    main()
