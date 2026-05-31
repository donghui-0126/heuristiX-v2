# Brandimarte FJSSP benchmark loader

This directory is the drop-in location for the canonical Brandimarte
instances Mk01–Mk10 (from *Routing and scheduling in a flexible job shop
by tabu search*, Brandimarte 1993, **Annals of Operations Research**).

## Adding the official instances

Place the files here as plain text named `Mk01.txt` … `Mk10.txt`.

```
data/brandimarte/
├── Mk01.txt
├── Mk02.txt
├── ...
└── Mk10.txt
```

The files can be obtained from any standard FJSSP benchmark archive
(e.g. operations-research course repositories or the supplementary
material of FJSSP papers). The parser expects the original Brandimarte
text format.

## File format

```
<n_jobs> <n_machines> <avg_eligibility>
<job_1>
<job_2>
...
<job_n>
```

Each job line:

```
<n_ops>  <op_1>  <op_2>  ...  <op_n>
```

Each op spec:

```
<n_eligible_machines>  (<machine_id> <proc_time>)+
```

* Machine IDs are **1-indexed** in the file; the loader normalises them
  to 0-indexed internally.
* When an op has multiple eligible machines with different processing
  times, the loader currently uses the **minimum** processing time as
  the canonical `Operation.processing_time`. Per-(op, machine) time
  matrices for full FJSSP fidelity is a separate TODO.

`sample_2x3.txt` is a tiny hand-crafted instance used by the parser
unit tests.

## CLI usage

```
./target/release/heuristix \
    --instance-file data/brandimarte/Mk01.txt \
    --scenario S1 --rule FIFO \
    --part-delay-ratio 0.2 --part-delay-k 1.0 \
    --instance-tightness 1.5 --ddt 1.0
```

When `--instance-file` is set the synthetic generator is skipped;
`--jobs`/`--machines`/`--flexibility` are ignored (the file decides
those). Due dates are synthesised as

```
due_j = release_j + instance_tightness × ddt × total_processing_j
```

per 실험설계서_수정 §7-1 (TWK + DDT).
