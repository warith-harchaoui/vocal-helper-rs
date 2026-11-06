# Vocal Helper (Rust)

[🇫🇷](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/LISEZMOI.md) · [🇬🇧](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/README.md)

[![crates.io](https://img.shields.io/crates/v/vocal-helper-rs.svg)](https://crates.io/crates/vocal-helper-rs) [![Licence : BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE)

Réécriture en Rust des chemins Nemotron gratuits et locaux de [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper) : la diarisation en flux avec Nemotron 3 Diarization puis la transcription avec Nemotron 3.5, chacune par son graphe ONNX. Pas un portage ligne à ligne : du Rust idiomatique de bout en bout.

## Pourquoi ce dépôt existe, dit honnêtement

**Pas pour la vitesse.** C'est mesuré, pas supposé. En profilant la diarisation sur deux minutes
d'audio, l'appel au graphe et le front-end chronométrés à part :

| | temps | part |
|---|---:|---:|
| graphe ONNX *(C++, identique depuis Rust)* | 90,47 s | **99,8 %** |
| front-end log-mel *(numpy, FFT en C)* | 0,06 s | 0,1 % |
| comptabilité du cache + Python | 0,10 s | 0,1 % |

Le plafond d'un portage Rust est de **0,2 % du temps** — un facteur temps réel de 0,754 au lieu de
0,755. La caisse `ort` appelle la même bibliothèque C++ ; les morceaux de Python sont soit
vectorisés, soit du travail sur des tableaux de 264 × 512 à raison d'un appel par seconde environ.
Qui promet un gain de RTF ici n'a pas mesuré.

**Pour des endroits où tourner.** Sur iOS et Android, le chemin Python n'existe pas : on n'embarque
pas raisonnablement CPython et une pile scientifique dans une application mobile. C'est un argument
de *possibilité* et non de performance : il est décisif. S'y ajoutent un binaire unique par
plateforme, aucune résolution de dépendances à l'installation, aucun interpréteur à démarrer,
sans oublier une empreinte mémoire modeste pour un service qui écoute en arrière-plan.

Même moteur, mêmes nombres, même vitesse — quelque part où Python n'accède pas.

## Portée

Tout ce qui vit **autour** du graphe est là : c'est l'essentiel du travail. Ce qui n'est pas
encore écrit est nommé comme tel plutôt que sous-entendu.

| module | ce qu'il fait | état |
|---|---|---|
| `mel` | le front-end log-mel que le graphe attend, à la trame près | **fait**, parité prouvée |
| `cache` | la tenue du cache de locuteurs et de la file d'un pas à l'autre | **fait**, parité prouvée |
| `schedule` | le calendrier de flux et les deux plafonds qui le bornent | **fait**, parité prouvée |
| `segments` | des probabilités par trame aux tours de parole | **fait**, parité prouvée |
| `accel` | choisir un moteur d'exécution d'après des mesures prises sur l'appareil | **fait** et appliqué par les deux sessions : l'appelant, puis `VOCAL_HELPER_PROVIDER`, puis une mesure sur l'appareil ; repli sur le CPU avec un avertissement ; features `cuda`, `tensorrt`, `coreml`, `directml`, `openvino` |
| `asr` | l'hôte de la transcription Nemotron 3.5 : front-end NeMo, fenêtres, CTC glouton, mots horodatés, balises de locuteur et d'événement | **fait**, parité prouvée sur les métadonnées et les sorties du vrai graphe |
| `onnx_meta` | lire les `metadata_props` d'un graphe sans charger ses poids | **fait** |
| liaison ONNX Runtime | piloter le graphe lui-même, par `ort`, derrière la feature `onnx` | **transcription faite** (`asr_session`, `examples/transcribe.rs` ; 4,4 % de WER d'écart au jumeau Python en int8 sur 20 min d'AMI, soit le bruit propre au graphe int8 (le jumeau Python contre lui-même, optimisations du graphe coupées : 4,4 %) ; mel identique à 1,3e-6 près ; RTF 0,05 sur CPU contre 0,022 à 0,028 pour le jumeau Python ; mesuré le 2026-11-05 sur un Apple M2 Max, `tests/jumeau_ami.rs`) ; **diarisation faite** (`diar_session`, `examples/diarize.rs` ; probabilités à 1,7e-6 du jumeau Python sur 3 min d'AMI, compression du cache comprise ; 1,9e-6 sur une autre minute d'AMI, remesuré pour cette version) |
| chemins payants | Gladia et pyannoteAI, en lot et en direct | pas écrits |

```sh
cargo test                                  # 116 tests, parité comprise
cargo clippy --all-targets -- -D warnings
```

La vérification de bout en bout contre le jumeau Python demande le graphe int8 et le
sous-corpus AMI que `vocal-helper` met en cache : elle est donc ignorée par défaut. Une
commande la relance :

```sh
VH_NEMOTRON_ASR_GRAPH=~/.cache/vocal-helper/nemotron-onnx/nemotron-asr-ctc.int8.onnx \
VH_AMI_SUBSET=~/.cache/vocal-helper/ami-subset \
cargo test --release --features onnx --test jumeau_ami -- --ignored --nocapture
```

Elle transcrit 20 minutes d'AMI et échoue au-delà de 6 % de WER contre les transcriptions
Python gardées dans `tests/fixtures/ami_jumeau_int8.json` (ou au-delà de 15 % sur une seule
minute). L'écart n'est pas nul et ne doit pas l'être : deux builds d'ONNX Runtime arrondissent
différemment les noyaux int8, si bien que les trames où deux jetons sont presque à égalité
basculent. `scripts/make_ami_jumeau.py` réécrit la fixture.

## Installation et usage

```toml
[dependencies]
vocal-helper-rs = { version = "0.2", features = ["onnx"] }
```

Sans `onnx`, la caisse se réduit à sa partie pure (front-ends, cache, calendrier, segments,
décodage CTC) : elle se compile en Rust 1.82 sans aucune dépendance native. Avec `onnx`, elle tire
`ort` 2.0.0-rc.15, qui demande Rust 1.92 et télécharge un ONNX Runtime précompilé à la
compilation.

Une feature par moteur d'exécution, chacune impliquant `onnx`, chacune liée aux runtimes
précompilés qu'`ort` sait télécharger : une feature compilée hors de sa plateforme échoue dans le
script de compilation d'`ort-sys`, pas à l'exécution.

| feature | moteur | plateformes dotées d'un runtime précompilé |
|---|---|---|
| `onnx` | CPU seul | toutes |
| `cuda` | CUDA 13 + cuDNN 9 | Linux x86_64 et aarch64, Windows x86_64 |
| `tensorrt` | TensorRT 10, puis CUDA | comme `cuda` (même téléchargement) |
| `coreml` | CoreML | macOS, iOS |
| `directml` | DirectML | Windows |
| `openvino` | OpenVINO | Linux x86_64, Windows x86_64 ; pas avec `cuda` |

**Les graphes ne sont pas livrés avec la caisse**, ils pèsent des centaines de mégaoctets : on
passe leur chemin. Le graphe de diarisation est l'export ONNX en un fichier de
[nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) fait pour
[Vernacula](https://github.com/christopherthompson81/vernacula), dont les métadonnées portent le
calendrier de flux et l'empreinte de silence ; c'est le fichier `nemotron3/nemotron-3-diarization.onnx`
du lot de poids que télécharge le jumeau Python `vocal-helper`.

```rust
use std::path::Path;
use vocal_helper_rs::diar_session::NemotronDiar;

// Échantillons f32 mono à 16 kHz ; toute autre fréquence est une erreur, pas un rééchantillonnage muet.
let mut diar = NemotronDiar::open(Path::new("nemotron-3-diarization.onnx"), Some(1.04))?;
for tour in diar.diarize(&echantillons, 16_000, 0.5)? {
    println!("{:.2}\t{:.2}\tSPK{}", tour.start, tour.end, tour.speaker);
}
```

`examples/diarize.rs` et `examples/transcribe.rs` font la même chose depuis un fichier WAV :

```sh
cargo run --release --features onnx --example diarize -- nemotron-3-diarization.onnx clip.wav
cargo run --release --features onnx --example transcribe -- <graphe-asr.onnx> clip.wav
```

Les deux attendent du 16 kHz et moyennent plusieurs canaux en mono. `diarize` écrit un tour par
ligne ; avec `--probs <sortie.f32>`, il écrit aussi les probabilités à 10 ms en `f32` petit-boutiste.

## Un portage ne vaut que sa parité

Le graphe ne prend pas de l'audio, il prend des trames log-mel. Sa configuration en donne la
charpente — 16 kHz, fenêtre de 400 échantillons, saut de 160, transformée sur 512, pré-accentuation
0,97, 128 bandes — mais pas les détails qui décident du résultat. Ils ont été **retrouvés en
comparant à l'extracteur du modèle**, pas devinés. Deux comptent :

- La fenêtre d'analyse de 400 points est **centrée** dans la transformée de 512 et non collée au
  début. La coller multiplie par seize l'écart aux sorties officielles — 0,530 au lieu de 0,033 —
  parce que le décalage de phase introduit change le spectre.
- Le plancher du logarithme est `2^-24`, lu dans le minimum des sorties officielles : −16,36, qui
  en est exactement le logarithme.

Les deux chemins alimentent le **même graphe** : s'ils divergent, le chemin mobile fabrique
silencieusement d'autres probabilités ; aucun test unitaire de chaque côté ne le verrait.
`tests/parite_mel.rs` compare donc à une référence produite par
`vocal_helper.nemotron.log_mel` sur quatre signaux choisis pour ce qui peut différer : un bruit
riche, un ton pur sur la charnière de l'échelle Slaney, deux tons éloignés pour détecter une fuite
entre filtres voisins ; enfin du silence, où le plancher doit être atteint exactement. L'accord est à
`1e-3` près.

Cela fait une chaîne de deux maillons : extracteur officiel → Python (0,006 d'écart moyen) → Rust
(≤ 1e-3).

## Accélérations : mesurer l'appareil, ne pas croire le raisonnement

Le raisonnement évident sur les accélérateurs est faux : on l'a payé. ONNX Runtime n'a pas de
moteur Metal, donc sur Apple Silicon l'accélération passe par CoreML ; le mettre en tête semblait
aller de soi. Mesuré sur le graphe de diarisation, CoreML donne un facteur temps réel de **1,257**
— plus lent que le direct, donc inutilisable — contre **0,528** pour le CPU seul. Le journal de la
session dit pourquoi : CoreML ne prend que 1193 nœuds sur 2827 et les découpe en **316
partitions**, si bien que chaque pas de flux paie autant de traversées entre deux moteurs.

Le second réglage « évident » était fautif aussi : forcer huit fils sur une machine à huit cœurs
performants mesure 0,953 contre 0,528 en laissant le runtime choisir.

La leçon n'est pas que CoreML est mauvais — il sera excellent sur un autre graphe ou dans une
version qui partitionne mieux. La leçon est qu'un ordre codé en dur sera faux quelque part, donc la
politique de `accel` choisit d'après **des mesures prises sur l'appareil** (`choose`) ;
un `Verdict` est attaché à l'empreinte du graphe (`applies_to`), parce que le découpage dépend des
opérateurs donc du modèle : c'est précisément l'erreur qu'un ordre codé en dur commet.

La politique est pure — pas de modèle, pas de réseau, pas d'ONNX — donc entièrement testable.

Les deux sessions l'appliquent, dans l'ordre d'autorité du jumeau Python :

1. l'argument `providers` de `NemotronAsr::open_with` ou de `NemotronDiar::open_with` ;
2. sinon `VOCAL_HELPER_PROVIDER` : un nom (`cuda`, `tensorrt`, `coreml`, `directml`, `openvino`,
   `cpu`…) ou un ordre (`tensorrt,cuda`). Un nom inconnu est une erreur : une faute de frappe ne
   doit pas passer pour CUDA pendant que le CPU tourne ;
3. sinon, quand plus d'un moteur est compilé, une **mesure sur l'appareil** : chacun reçoit 5 s
   de chauffe puis 30 s d'audio synthétique chronométrées, le plus rapide gagne, puis le
   `Verdict` est gardé sous l'empreinte du graphe dans `$VOCAL_HELPER_RS_CACHE` (par défaut
   `~/.cache/vocal-helper-rs`). Avec `onnx` seul, seul le CPU est compilé : rien n'est mesuré.

La mesure se paie une fois par graphe et par machine ; elle n'est pas gratuite pour autant : sur un M2 Max
avec `coreml`, la première ouverture du graphe de diarisation a pris une minute environ ; elle a
retrouvé ce qui est écrit plus haut (CPU 0,74 contre CoreML 1,12 sur cette passe). Pour le graphe
ASR, les deux se valaient (0,084 contre 0,086). Épingler un moteur évite la mesure.

Un moteur qui échoue à s'enregistrer — pas de GPU, mauvaise version de CUDA, feature absente — est
sauté avec un avertissement sur la sortie d'erreur, puis le suivant est essayé ; le CPU ferme
toujours la marche. Une session qui ne se construit pas sur les moteurs retenus est reconstruite
sur le CPU seul. L'ouverture n'échoue jamais faute d'accélérateur. À la différence du jumeau
Python, un moteur connu mais absent de la compilation est sauté avec cet avertissement plutôt que
refusé.

## Ubuntu + NVIDIA

```toml
vocal-helper-rs = { version = "0.2", features = ["cuda"] }        # ou "tensorrt"
```

`ort` 2.0.0-rc.15 télécharge **ONNX Runtime 1.31.0** construit par pyke pour **CUDA 13 seulement** :
il n'existe pas de construction CUDA 12 à télécharger (`ORT_CUDA_VERSION=12` fait échouer la
compilation plutôt que de se rabattre). ONNX Runtime est lié statiquement dans le binaire ; les
moteurs restent des bibliothèques à part, que `cargo build` lie à côté dans `target/<profil>/` :
`libonnxruntime_providers_shared.so`, `libonnxruntime_providers_cuda.so` puis, pour `tensorrt`,
`libonnxruntime_providers_tensorrt.so`. **Livrez-les à côté de l'exécutable.**

Ce dont la machine a besoin, lu par `ldd` sur ces bibliothèques dans la CI de ce dépôt :

| feature | bibliothèques cherchées à l'exécution | d'où |
|---|---|---|
| `cuda` | `libcuda.so.1` | le pilote NVIDIA, branche 580 ou plus récente (ce que CUDA 13 exige) |
| `cuda` | `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libcurand.so.10` | les bibliothèques d'exécution de CUDA 13 |
| `cuda` | `libcudnn.so.9` et ses sous-bibliothèques | cuDNN 9 construit pour CUDA 13 (exigé par la liste de préchargement d'`ort`, sans être un lien direct du moteur CUDA) |
| `tensorrt` | `libnvinfer.so.10`, `libnvonnxparser.so.10` | TensorRT 10 pour CUDA 13 |

Sur Ubuntu avec le dépôt apt de NVIDIA, `sudo apt install cuda-libraries-13-0 libcudnn9-cuda-13`
couvre `cuda` (ajouter `tensorrt` pour le moteur TensorRT). Pour vérifier une machine :

```sh
nvidia-smi                                   # pilote chargé, « CUDA Version: 13.x » ou plus
ldconfig -p | grep -E 'libcudart.so.13|libcublas.so.13|libcudnn.so.9'
ldd target/release/libonnxruntime_providers_cuda.so | grep 'not found'   # ne doit rien afficher
VOCAL_HELPER_PROVIDER=cuda cargo run --release --features cuda --example transcribe -- graphe.onnx clip.wav
```

La dernière ligne affiche `moteurs [Cuda, Cpu]` quand CUDA tourne vraiment. **Rien n'est perdu
quand il ne tourne pas** : un moteur qui échoue à s'enregistrer (pas de GPU, pilote trop ancien,
bibliothèques CUDA 12, cuDNN absent) est sauté avec une ligne sur la sortie d'erreur qui cite
l'erreur d'ONNX Runtime lui-même, puis la session s'ouvre sur le CPU. `providers()` dit, sur
chacune des deux sessions, ce qui a été retenu.

**Ce qui est vérifié, ce qui ne l'est pas.** La CI compile et teste la caisse sur `ubuntu-latest`
avec `--features cuda,tensorrt` : la construction CUDA d'ONNX Runtime est téléchargée puis liée,
et chaque session retombe sur le CPU sur ce runner, qui n'a pas de GPU. Aucun runner gratuit n'en
a : **l'inférence sur un vrai GPU NVIDIA n'a pas été lancée par ce dépôt**. Les noms de
bibliothèques ci-dessus viennent des bibliothèques elles-mêmes, pas de cette exécution.


## D'où viennent ces chiffres

La campagne de mesure derrière ces choix est consignée dans
[`doc/diarization-results.fr.md`](https://github.com/warith-harchaoui/vocal-helper/blob/main/doc/diarization-results.fr.md)
de vocal-helper, impasses comprises : quatre paramétrages de seuil réfutés, le regroupement sans
centroïde fermé, enfin la découverte que la référence d'AMI ne peut pas arbitrer parce qu'elle est
transcrite sur les micros-casques alors qu'on donne au moteur le mixage de salle.

## Skill d'agent

`skills/vocal-helper-rs/` emballe cette caisse en **Claude Skill** comme en **skill OpenCode** : un
agent peut la découvrir puis écrire du Rust avec elle pour le compte de l'utilisateur.
L'installation pour les deux hôtes est dans
[`skills/README.md`](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/skills/README.md).

Son catalogue de déclencheurs est vérifié, pas décoratif :
[`TRIGGERS.md`](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/TRIGGERS.md) est le
catalogue tourné vers l'utilisateur, `references/triggers.md` son miroir tourné vers l'agent.
`tests/skill.rs` fait échouer `cargo test` si les deux divergent, si la `description` du skill
dépasse le plafond de 1024 caractères ou perd sa clause SKIP, voire si `SKILL.md` cite une
référence absente. Le hook pre-push le lance ; `scripts/install-hooks.sh` active ce hook une fois
par clone.

## Voisinage

Fait partie du même outillage local-first que [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper) (Python), [`audio-helper-rs`](https://github.com/warith-harchaoui/audio-helper-rs), [`capture-helper-rs`](https://github.com/warith-harchaoui/capture-helper-rs), [`podcast-helper-rs`](https://github.com/warith-harchaoui/podcast-helper-rs), [`youtube-helper-rs`](https://github.com/warith-harchaoui/youtube-helper-rs) et la suite [AI Helpers](https://github.com/warith-harchaoui/ai-helpers). Réécriture indépendante, pas une liaison.

## Licence

BSD-3-Clause, voir [LICENSE](LICENSE). Les poids de modèle qu'elle charge ne sont pas les nôtres et
portent leurs propres termes — Nemotron 3 Diarization est à NVIDIA, sous OpenMDW License 1.1.

## Auteur

[Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui)
