# FlagOS Season 2, Track 2: measured MiniCPM5-2B optimisation

Preparation code for the [official Track 2 challenge](https://www.kaggle.com/competitions/flag-os-open-computing-global-challenge-track-2).
The target is MiniCPM5-2B served through the fixed `vllm-plugin-FL` competition
branch and FlagGems v5.3.5. The published launcher and operator checks extend
the author's controlled-kernel evaluation workflow used in
[Track 1](https://github.com/Agnuxo1/FlagOS-Track1-S2).

## Actual state, 28 October

**The RTX3090 development baseline completed.** The official benchmark script
ran both throughput cases four times: one initial run and three runs used by
its fixed `SKIP_FIRST` summary rule. All eight raw runs completed every request.

| Prefill / decode / concurrency / requests | Mean output tokens/s | Mean TTFT (ms) |
|---|---:|---:|
| 4096 /1024 /64 /256 | **1190.38** | 4236.90 |
| 16384 /1024 /64 /128 | **238.61** | 152391.38 |

[All eight raw rows and the upstream summary](evidence/rtx3090-baseline-20261028)
are preserved, including the slower initial runs. The workload uses random
benchmark tokens, not an accuracy dataset. A shutdown `EngineDeadError` follows
SIGTERM after these completed measurements; it is recorded in provenance.
No throughput improvement, accuracy qualification or official score is claimed.
CUDA is a development proxy; official BI150/C500 measurements are still needed.

The current team plan targets **Track 1**. Track 2 stays prepared while the
one-track/team restrictions and official hardware allocation are resolved.
This public code is not a second competition registration or ZIP submission.

## Applied reliability improvements

- The launcher fails if its port is already occupied, the process exits, the
  model is not ready or the official benchmark fails. A final echo cannot hide
  a failed benchmark anymore.
- Readiness requires an HTTP-success response and the expected model ID.
- Logs, exact software versions and GPU information are preserved; cleanup
  addresses only the process launched by this script.
- The operator harness checks numerical correctness before timing. Its fused
  in-place case includes input clones in the measured cost, explicitly a
  call-level measurement rather than an isolated-kernel latency.

## Three prespecified experiment directions

| Arm | Single change | Required evidence |
|---|---|---|
| T2-E1 | Residual-add/RMSNorm tile geometry | Three seeds and launches, correctness, both official throughput cases |
| T2-E2 | SiLU/multiply launch geometry | Same model, dtype and arguments; no operator-selection bypass |
| T2-E3 | Fixed combination of individually qualified kernel changes | Compare with each single arm and upstream baseline |

These are **pending experiments**, not measured gains. Keep all failed cases
and regressions. Require Math500 Level3 accuracy >=0.95, TTFT no worse than
baseline+1%, and throughput improvement above the official 1% noise band.
No quantisation, speculative decoding, benchmark edits, model substitution or
serving-parameter tuning is introduced.

## Reproduce

The launcher uses the official Iluvatar argument profile. MetaX requires its
separate official profile (without the compilation-config argument), not this
launcher unchanged. Pin the official `flagos-2026-s2` branch (`13eb9be` at preparation) and
FlagGems v5.3.5, obtain the required model and approved hardware, then set
`FLAGOS_PROJECT_ROOT` to the isolated environment containing them:

```sh
bash bench/run_bench.sh E0_baseline_r1 '[[4096,1024,64,256],[16384,1024,64,128]]'
python bench/micro_ops.py --tag E0
```

Do not use a local GPU as an automatic fallback. Queue compatible proxy jobs
on servidor-ia and keep their scope distinct from official measurements.

## Dates, sources and licence

The [checked task contract](evidence/official-contract-20261028.md) gives
registration/compute application closing10November and technical delivery
20November23:59UTC+8 (16:59Madrid). The later Kaggle container date does not
extend that technical deadline. A compute application also needs the required
team/CV information; none is invented here.

Own harness code: Apache-2.0. Upstream FlagGems, vLLM, the model and
`vllm-plugin-FL` retain their own licences. Their source, weights, credentials
and competition data are not redistributed by this repository.
