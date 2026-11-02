# Resuming the #321 numerics gate

State when this checkpoint was written:

- granite-4.0-1b-4bit: harness check done; baseline runs 0/5, candidate runs 0/5; reference done; checks 2-3 not run; check 1 not run.

Every phase skips outputs that already exist, so after an interruption (including a power loss) the same commands continue where the run stopped. The host layout is `~/gpu/upr` (`scripts/setup.sh` rebuilds the wheels and venvs from the named commits; `scripts/ref_setup.sh` rebuilds the reference venv with the hash checks). The scripts run from that layout: `scripts/gate_tf.py`, `perop.py`, and the receipt's `scripts/ab.py`, `once.py`, `prov.py`, `mlx_provenance.py` go in `~/gpu/upr/probes/`; `run_gate.sh`, `runner316.sh`, `analyze_gate.py`, `same_outputs.py` and the text in `~/gpu/upr/gate/`. Copy this directory's records back to `~/gpu/upr/gate/out/` first if the host lost them (`runs/<pr>-<model>/*.json.gz` to `<pr>-<model>/*.json`, `ref-*.json.gz` to `ref-*.json`, `ref-parts/*.json.gz` to `ref-parts/*.json`, `teacher-*.json` as they are).

```
cd ~/gpu/upr/probes
bash ../gate/run_gate.sh ref "granite-4.0-1b-4bit"   # reference, skips finished segments
bash ../gate/run_gate.sh arms321      # runs only the missing pairs (five runs per arm, main against p6)
bash ../gate/run_gate.sh perop321     # check 1 (skips finished outputs)
bash ../gate/run_gate.sh analyze321   # checks 1-3 scores and compare_tf9
```

Or `setsid nohup bash ~/gpu/upr/gate/runner_all.sh > ~/gpu/upr/logs/gate-runner-all.log 2>&1 &` for the whole gate in order (#316, #317, #321), skipping what is done.
