# NCU report skill: Ampere, Ada and Blackwell RTX

This repository-local skill covers SM80/SM86/SM89/SM120 on Windows and Linux.
RTX 30x0/40x0/50x0 correspond to SM86/SM89/SM120; SM80 covers A100 separately.

Start at [SKILL.md](SKILL.md). It links three architecture guides, ten workflow
references and [helpers](helpers/README.md). The pipeline is capability
discovery → build → reference verification → unprofiled benchmark → targeted
collection → action-aware analysis → report → controlled experiment.

The architecture guides cite the supplied NVIDIA GA102, Ada and RTX Blackwell
whitepapers and CUDA documentation. Installed NCU help and actual reports
determine section/metric availability.

Report analysis needs Python and a compatible NCU `extras/python` module.
Collection also needs a supported GPU/driver and counter access. No fixed CUDA
or NCU version is assumed. Set `NCU_PYTHON_PATH` to the matching `extras/python`
directory when automatic discovery cannot locate it.

Source harnesses go under `csrc/tests/<family>/`, build outputs under `build/`,
and reports under a fresh `profile/<run>/`.
