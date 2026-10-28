# FlagOS Open Computing Global Challenge S2 – Track 2 (official rules snapshot)

Source: https://www.kaggle.com/competitions/flag-os-open-computing-global-challenge-track-2
Captured: 2026-10-28 (browser get_page_text, logged-in Kaggle session of Agnuxo1).
Kaggle banner: "Your submission is due by Dec 30, 2026 at 5:00 PM GMT+1" (Kaggle writeup container; the
binding deadline for the work is the FlagOS-platform one below).

## Task
Track 2: Inference Throughput Performance Optimization for MiniCPM5-2B. Prize pool USD 9,000 (4,500/3,000/1,500).
Framework: vllm-plugin-FL branch `flagos-2026-s2`; FlagGems tag `v5.3.5`.
Hardware: Iluvatar CoreX BI150 and MetaX C500 (64 GB), officially provided compute only
(Iluvatar via https://flagos.io/OnlineLab after approval; MetaX via compute voucher on https://ai.gitee.com/compute).
Model: OpenBMB/MiniCPM5-2B (ModelScope). Only this model may be used.

## Schedule (UTC+8) -> Europe/Madrid (CET, UTC+1 after 25-10-2026)
- Registration: 19-07-2026 – 10-11-2026 -> ends 10-11-2026 16:59 Madrid (assuming 23:59 UTC+8)
- Compute application: 14-09-2026 – 10-11-2026 (proposal "TeamName.pdf" + team intro with CVs; 7 business-day review; first come first served)
- Compute review/allocation: 21-09 – 15-11-2026
- Development & submission: 21-09 – 20-11-2026
- Submission deadline: 23:59 20-11-2026 UTC+8 -> 16:59 20-11-2026 Madrid
- Evaluation: early Dec 2026; results mid-Dec 2026
- Winners: PR to flagos-ai/vllm-plugin-FL@flagos-2026-s2 within 3 business days after evaluation.

## Submission
`TeamName.zip` = TeamName/{FlagGems, readme.md, report.pdf, vllm-plugin-FL}, uploaded at
https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn . Registration at
https://flagos.io/race-detail-season2?id=8q4m2x7p&lang=cn (and https://flagos.io/Register).

## Evaluation
- vllm serve fixed args (Iluvatar: --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}', --gpu-memory-utilization 0.85, --max-model-len 131072; MetaX same minus compilation-config).
- Accuracy: evalscope math_500 Level 3, temperature 1.0, top_p 0.95, max_tokens 32768, batch 8. Baseline 0.962, must be >= 0.95.
- Performance: total tokens/s with vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py, cases [[4096,1024,64,256],[16384,1024,64,128]]. TTFT must not exceed baseline +1%; throughput within 1% = noise; a scenario scores only above +1%; score = mean of 4k and 16k.
- Total = perf(Iluvatar) + perf(MetaX) + innovation; perf = 70% x improvement ratio; innovation = 30% x expert score of report.

## Prohibited
Changing vllm serve params; quantization or speculative decoding via hyperparameters; benchmark-script tuning;
switching operators / deleting operator-selection logic without operator optimization; merging newer upstream branch code;
other models. Results invalid on accuracy drop, abnormal output, behaviour change, prohibited strategies, non-reproducibility.
