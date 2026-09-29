# Teacher label verification and representation hypothesis

audit_teacher_contract.py evaluated the original deterministic learned teacher
on all5400 saved public observations from the9 FIT successes. Each episode's
MSE was below2.1e-13; maximum absolute action difference across episodes was
3.58e-6. The original checkpoint hash and network structures are recorded in
teacher_contract_audit.json. Small differences are consistent with batched
versus individual floating-point execution. No model weights changed.

Teacher: public119 -> ReLU256 -> ReLU256 -> four squashed actions,98,568 actor
parameters including distribution head. Student: public119 -> LSTM128 ->
Tanh128 -> four means,144,516 actor parameters (excluding log_std). Therefore
the student is not simply smaller by total parameter count. The paths and
nonlinearities differ, and the student has no direct observation skip path.

This rules out a large source-label mismatch for these FIT examples. It does
not establish that observations suffice for every possible trajectory, that
the teacher generalizes well, or that a skip connection guarantees learning.
The teacher's historical full-success rate remained11/2048, with9 FIT successes.

Proposed next controlled test: zero-initialized trainable public-observation
bypass into the recurrent actor. First verify exact starting behaviour, correct
episode resets and save/load parity, including forward/evaluate_actions used
by RL (not just predict). Only then compare bounded fitting while retaining
single-pass tests. Keep one learned policy, all119 inputs public, no simulator
state, scripted actions or success-criteria changes. Avoid increasing run length
on the existing failed architecture without evidence.

Latest RL run remains stopped8192 for retention. No new training, physical flight,
reserved examples,7M acceptance data or publication during this audit.
