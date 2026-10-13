# Runbook: Stage 2 (60M model, multi-source data) on RunPod

Stage 1 is done (30M, TinyStories, 100M tokens, val_loss 1.631 / ppl 5.11, measured; see
`reports/stage1_30m_tinystories/`). This runbook covers Stage 2 only.
Times and costs are estimates from measured speeds (60.6k tok/s for the 60M model on the RTX PRO 6000 MIG
24 GB slice used before). The data-build time is UNMEASURED except TinyStories (~2 min).

## Create the pod
- Any GPU with 24 GB or more, PyTorch 2.8 template, **container disk 40 GB**, one GPU. Note the EUR/h.
- Upload ONE file to /workspace: `tenslam-minilm-runpod.zip` (must be exactly **195795 bytes**).

## Paste each block ONCE
**Block A: setup (~2 min)**
```bash
cd /workspace && ls -l tenslam-minilm-runpod.zip && unzip -tq tenslam-minilm-runpod.zip | tail -1
rm -rf /tmp/x && unzip -q tenslam-minilm-runpod.zip -d /tmp/x && bash /tmp/x/scripts/pod_setup.sh 2>&1 | tail -8
```
Expect `tokenizer OK`, a GPU line, `SETUP DONE`.

**Block B: start everything (one command, then you can leave the terminal)**
```bash
cd /workspace/tenslam-minilm
export HF_TOKEN=PUT_YOUR_TOKEN_HERE
nohup bash scripts/pod_stage2.sh > stage2.log 2>&1 &
sleep 45; cat stage2.log
```
`stage2.log` should say `STEP 1/4: building the data mix`.
The script does: data build -> 100-step smoke test -> long training -> packaging. It stops with a clear message
if a step fails. If the pod or script is interrupted, run Block B again: it resumes.

**Block C: status check (as often as you like)**
```bash
cd /workspace/tenslam-minilm; tail -4 stage2.log; tail -2 prepare_mix.log; tail -2 train_mix.log 2>/dev/null; df -h /workspace | tail -1
```
What you will see over time:
1. `prepare_mix.log` prints one line per source, then the token-share table, `manifest -> ...`.
   Sources: tinystories, fineweb_edu, wikipedia, gutenberg, cosmopedia_stanford, cosmopedia_openstax.
2. `smoke_mix.log` ends with `finished`.
3. `train_mix.log`: loss starts near 9.7, `val_loss/<source>` every 500 steps, ~60k tok/s, ~4.6 h for 1B tokens (est.).
4. `stage2.log` ends with `ALL DONE`.

**Send me:** the output of Block C after the data build finishes (the share table), and again after the first
`val_loss` lines. If anything says FAILED, send the last 30 lines of the named log.

## When ALL DONE: download (the script already packaged and hashed everything)
In JupyterLab's file panel (folder `tenslam-minilm`) download:
- `stage2_results.tgz` (expected ~230 MB; the size is printed in stage2.log)
- `stage2_results.sha256`
- If the tgz download is truncated (size differs from stage2.log), download every `stage2.part_00`, `_01`, ... instead
  (20 MB each) and tell me. I join them and verify the hashes.

Put them in `/home/saroj/Aihackathon/`, tell me the sizes, and only then **terminate** the pod (not just stop).

## Cost guard (estimates, EUR 0.67/h)
Data build ~0.5-1.5 h + training ~4.6 h => ~EUR 3.5-4.5. `max_hours = 5.5` stops training at 5.5 h at the latest.
**Cut cost:** before Block B run `sed -i 's/max_tokens = 1000000000/max_tokens = 600000000/' configs/train_mix.toml`
(about 2.8 h of training, ~EUR 1.9, slightly weaker model).
The trainer's time limit does NOT stop the pod billing; terminate when done.

## Tokens
Use a read-only HF token, type it only in the pod terminal, and delete it at huggingface.co/settings/tokens afterwards.
