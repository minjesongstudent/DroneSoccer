# Mixed trial update audit

Read-only CUDA audit completed on saved mixed-run trajectories. No optimizer
steps were taken and source/final checkpoint hashes remained unchanged.

On all 2,011 first-batch public-input frames, with each actor reconstructing
its own memory, source-to-final Gaussian KL averaged 0.000008995 and peaked at
0.000112891. Mean deterministic action MSE was 0.00000017823, or RMS 0.000422
in normalized action units. Clipped-action MSE was almost identical. These
measurements show a very small update on these histories, not a guarantee of
small closed-loop trajectory changes or unchanged behavior elsewhere.

Per-flight gradients were measured at the original checkpoints for batches
0 and 2. Their summed norm divided by the sum of individual norms was 0.325
and 0.304. Mean pair cosine was 0.00043 and -0.00163; 55.2% and 47.6% of pairs
had negative cosine. Most directions are nearly orthogonal, not evidence of
uniform strong opposition. The audit alone cannot establish that gradient
noise caused the lack of improvement.

Full matrices and norms are preserved in the mixed run's
`gradient_audit/result.json`; the reproducible script is
`diagnostics/audit_mixed_gradients.py`.

## Next bounded test

The evidence supports considering a larger update constrained by measured
action-distribution change on authentic histories, rather than extending the
same four tiny updates. A proposed experiment should predeclare a KL bound,
rollback both parameters and optimizer state if exceeded, and retain the
fixed single-pass stop checks. Rebuild memory under each candidate's weights
before measuring KL; reusing source hidden states would invalidate that check.
Do not infer success from a larger gradient or changed actions: natural-start
repeat passages and crash rates remain decisive. Preserve the current source.
