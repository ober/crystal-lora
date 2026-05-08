#!/usr/bin/env python3
"""
Train Crystal LoRA on a RunPod GPU using axolotl.

Workflow (subcommands):
  up      — Create the GPU pod (axolotl image, SSH enabled), wait for SSH ready
  push    — scp the axolotl config + training data to the pod
  train   — Kick off `axolotl train` inside tmux on the pod (returns immediately)
  tail    — Tail the training log
  pull    — scp the merged model back to ./runpod-pipeline-final/
  down    — Terminate the pod
  status  — Show pod info + last training log line

State persisted to .runpod_state.json

Modes (CRYSTAL_MODE env var):
  pipeline (default)   — staged CPT → SFT → DPO on 1× A100 80GB
  lora                 — single-stage LoRA SFT on 1× A100 80GB
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
STATE_FILE = REPO / ".runpod_state.json"
TOKEN_FILE = Path.home() / ".runpod.token"

# Default to staged pipeline; override with CRYSTAL_MODE=lora for plain SFT.
MODE = os.environ.get("CRYSTAL_MODE", "pipeline")

if MODE == "pipeline":
    POD_NAME = "crystal-train-pipeline"
    GPU_COUNT = 1
    CONTAINER_DISK_GB = 200              # base + previous_merged + new_merged peak
    HOURLY_RATE = 2.10                   # 1× A100-SXM4 80GB community price
    LOCAL_CONFIG_NAME = None             # pipeline pushes its own per-stage configs
elif MODE == "lora":
    POD_NAME = "crystal-train-lora"
    GPU_COUNT = 1
    CONTAINER_DISK_GB = 100              # base model + cache + LoRA outputs
    HOURLY_RATE = 1.89                   # 1× A100 80GB PCIe community price
    LOCAL_CONFIG_NAME = "axolotl_crystal_sft.yaml"
else:
    sys.exit(f"Unknown CRYSTAL_MODE={MODE!r} (expected: pipeline, lora)")

# Pipeline mode uses SXM4 (PCIe community capacity is intermittent for long jobs).
GPU_TYPE = "NVIDIA A100-SXM4-80GB" if MODE == "pipeline" else "NVIDIA A100 80GB PCIe"
IMAGE_NAME = "axolotlai/axolotl-cloud-uv:main-py3.12-cu128-2.10.0"
VOLUME_GB = 0                            # no persistent volume; we scp out

# Files to push (lora mode only)
LOCAL_CONFIG = (REPO / LOCAL_CONFIG_NAME) if LOCAL_CONFIG_NAME else None
LOCAL_TRAIN_DATA = REPO / "training_data_together.jsonl"

REMOTE_WORKSPACE = "/workspace"
REMOTE_CONFIG = f"{REMOTE_WORKSPACE}/{LOCAL_CONFIG_NAME}" if LOCAL_CONFIG_NAME else None
REMOTE_TRAIN_DATA = f"{REMOTE_WORKSPACE}/data/train.jsonl"
REMOTE_LOG = f"{REMOTE_WORKSPACE}/train.log"
REMOTE_OUTPUT = f"{REMOTE_WORKSPACE}/output"
LOCAL_ADAPTER_DIR = REPO / "runpod-adapter"

# ─── Pipeline mode (staged CPT → SFT → DPO) ───────────────────────
PIPELINE_STAGES = [
    {
        "name": "cpt",
        "config_local": "axolotl_crystal_cpt.yaml",
        "data_local": "cpt_corpus.jsonl",
        "data_remote": f"{REMOTE_WORKSPACE}/data/cpt.jsonl",
        "output_dir": f"{REMOTE_WORKSPACE}/output_cpt",
        "merged_dir": f"{REMOTE_WORKSPACE}/cpt_merged",
        "needs_base_substitution": False,
        "previous_merged": None,
    },
    {
        "name": "sft",
        "config_local": "axolotl_crystal_sft.yaml",
        "data_local": "training_data_together.jsonl",
        "data_remote": f"{REMOTE_WORKSPACE}/data/sft.jsonl",
        "output_dir": f"{REMOTE_WORKSPACE}/output_sft",
        "merged_dir": f"{REMOTE_WORKSPACE}/sft_merged",
        "needs_base_substitution": True,
        "previous_merged": f"{REMOTE_WORKSPACE}/cpt_merged",
    },
    {
        "name": "dpo",
        "config_local": "axolotl_crystal_dpo.yaml",
        "data_local": "dpo_pairs.jsonl",
        "data_remote": f"{REMOTE_WORKSPACE}/data/dpo.jsonl",
        "output_dir": f"{REMOTE_WORKSPACE}/output_dpo",
        "merged_dir": f"{REMOTE_WORKSPACE}/dpo_merged",
        "needs_base_substitution": True,
        "previous_merged": f"{REMOTE_WORKSPACE}/sft_merged",
    },
]
PIPELINE_FINAL_DIR = REPO / "runpod-pipeline-final"


def load_token():
    if not TOKEN_FILE.exists():
        sys.exit(f"RunPod token not found at {TOKEN_FILE}")
    return TOKEN_FILE.read_text().strip()


def setup_runpod():
    import runpod
    runpod.api_key = load_token()
    return runpod


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2))


def ssh_target(state):
    host = state.get("ssh_host")
    port = state.get("ssh_port")
    if not host or not port:
        sys.exit("Pod SSH not yet ready. Run `up` first.")
    return host, port


def run_ssh(state, cmd, *, capture=False, check=True):
    host, port = ssh_target(state)
    full = [
        "ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        f"root@{host}", cmd,
    ]
    if capture:
        return subprocess.run(full, check=check, capture_output=True, text=True)
    return subprocess.run(full, check=check)


def run_scp(state, local, remote, *, recursive=False):
    host, port = ssh_target(state)
    args = ["scp", "-P", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
    if recursive:
        args.append("-r")
    args.extend([str(local), f"root@{host}:{remote}"])
    subprocess.run(args, check=True)


def scp_back(state, remote, local, *, recursive=False):
    host, port = ssh_target(state)
    args = ["scp", "-P", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
    if recursive:
        args.append("-r")
    args.extend([f"root@{host}:{remote}", str(local)])
    subprocess.run(args, check=True)


# ─── Subcommands ────────────────────────────────────────────────────

def cmd_up():
    runpod = setup_runpod()
    state = load_state()
    if state.get("pod_id"):
        print(f"Pod already exists: {state['pod_id']}. Run `down` first or delete .runpod_state.json")
        sys.exit(1)

    print(f"Creating pod (mode: {MODE}, GPU: {GPU_COUNT}× {GPU_TYPE}, image: {IMAGE_NAME})...")
    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=IMAGE_NAME,
        gpu_type_id=GPU_TYPE,
        cloud_type="ALL",
        gpu_count=GPU_COUNT,
        container_disk_in_gb=CONTAINER_DISK_GB,
        volume_in_gb=VOLUME_GB,
        support_public_ip=True,
        start_ssh=True,
        ports="22/tcp",
    )
    pod_id = pod["id"]
    state["pod_id"] = pod_id
    state["created_at"] = int(time.time())
    save_state(state)
    print(f"  pod_id: {pod_id}")

    print("Waiting for SSH (this can take 1-3 min while the image pulls)...")
    deadline = time.time() + 600
    while time.time() < deadline:
        info = runpod.get_pod(pod_id)
        runtime = info.get("runtime") if isinstance(info, dict) else None
        ports = (runtime or {}).get("ports") or []
        ssh_port = next((p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")), None)
        if ssh_port:
            state["ssh_host"] = ssh_port["ip"]
            state["ssh_port"] = ssh_port["publicPort"]
            save_state(state)
            print(f"  ssh: root@{state['ssh_host']}:{state['ssh_port']}")
            break
        time.sleep(10)
        print("  ...still waiting", flush=True)
    else:
        sys.exit("Timed out waiting for SSH")

    # Brief grace period for sshd to actually accept connections
    print("Waiting for sshd to accept connections...")
    for _ in range(30):
        try:
            run_ssh(state, "echo ready", capture=True, check=True)
            print("  ssh up.")
            break
        except subprocess.CalledProcessError:
            time.sleep(5)
    else:
        sys.exit("sshd not responding")

    print("\nNext: python3 runpod_train.py push")


def cmd_push():
    state = load_state()
    if MODE == "pipeline":
        return _pipeline_push(state)
    print("Creating workspace dirs on pod...")
    run_ssh(state, f"mkdir -p {REMOTE_WORKSPACE}/data {REMOTE_OUTPUT}")
    print(f"Uploading {LOCAL_CONFIG.name}...")
    run_scp(state, LOCAL_CONFIG, REMOTE_CONFIG)
    print(f"Uploading training data ({LOCAL_TRAIN_DATA.stat().st_size//(1024*1024)}MB)...")
    run_scp(state, LOCAL_TRAIN_DATA, REMOTE_TRAIN_DATA)
    print("\nNext: python3 runpod_train.py train")


def cmd_train():
    state = load_state()
    if MODE == "pipeline":
        return _pipeline_run(state)
    cmd = (
        f"cd {REMOTE_WORKSPACE} && "
        f"tmux new-session -d -s train "
        f"\"export PATH=/workspace/axolotl-venv/bin:\\$PATH; "
        f"axolotl train {REMOTE_CONFIG} 2>&1 | tee {REMOTE_LOG}; "
        f"echo TRAINING_DONE >> {REMOTE_LOG}\""
    )
    print("Kicking off training in tmux session 'train'...")
    run_ssh(state, cmd)
    state["training_started_at"] = int(time.time())
    save_state(state)
    print("\nMonitor: python3 runpod_train.py tail")
    print("(or:     python3 runpod_train.py status)")


def cmd_tail():
    state = load_state()
    host, port = ssh_target(state)
    log = REMOTE_LOG if MODE != "pipeline" else f"{REMOTE_WORKSPACE}/cpt.log"
    print(f"Tailing {log} (Ctrl-C to stop)...\n")
    subprocess.run([
        "ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        f"root@{host}", f"tail -F {log}",
    ])


def cmd_status():
    runpod = setup_runpod()
    state = load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod. Run `up` first.")
        return
    info = runpod.get_pod(pod_id)
    print(f"Pod {pod_id}: {info.get('desiredStatus','?')}")
    runtime = info.get("runtime") or {}
    if runtime.get("uptimeInSeconds"):
        print(f"  uptime: {runtime['uptimeInSeconds']//60} min")
    if state.get("ssh_host"):
        print(f"  ssh: root@{state['ssh_host']}:{state['ssh_port']}")
    if MODE == "pipeline":
        for stage in PIPELINE_STAGES:
            r = run_ssh(state, f"cat {REMOTE_WORKSPACE}/{stage['name']}.done 2>/dev/null", capture=True, check=False)
            mark = r.stdout.strip() or "pending"
            print(f"  [{stage['name']}] {mark}")
    if state.get("training_started_at"):
        log = REMOTE_LOG if MODE != "pipeline" else f"{REMOTE_WORKSPACE}/cpt.log"
        try:
            r = run_ssh(state, f"tail -3 {log} 2>/dev/null || echo '(no log yet)'", capture=True, check=False)
            print("\nRecent log:")
            for line in r.stdout.splitlines():
                print(f"  {line}")
        except Exception as e:
            print(f"  (log fetch failed: {e})")


def cmd_pull():
    state = load_state()
    if MODE == "pipeline":
        return _pipeline_pull(state)
    LOCAL_ADAPTER_DIR.mkdir(exist_ok=True)
    print(f"Looking for adapter on pod...")
    r = run_ssh(state, f"ls -d {REMOTE_OUTPUT}/* 2>/dev/null | tail -5", capture=True)
    print(r.stdout)
    print(f"Downloading {REMOTE_OUTPUT}/ -> {LOCAL_ADAPTER_DIR}/")
    scp_back(state, f"{REMOTE_OUTPUT}/.", LOCAL_ADAPTER_DIR, recursive=True)
    print(f"Downloaded. Files:")
    for p in sorted(LOCAL_ADAPTER_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(LOCAL_ADAPTER_DIR)}  ({p.stat().st_size//1024}K)")


# ─── Pipeline mode helpers ──────────────────────────────────────────

def _pipeline_push(state):
    """Push all stage configs and datasets in one shot."""
    print("Creating workspace dirs on pod...")
    run_ssh(state, f"mkdir -p {REMOTE_WORKSPACE}/data {REMOTE_WORKSPACE}/cache")
    for stage in PIPELINE_STAGES:
        cfg = REPO / stage["config_local"]
        if not cfg.exists():
            sys.exit(f"missing {cfg} — run the build_*.py scripts first")
        data = REPO / stage["data_local"]
        if not data.exists():
            sys.exit(f"missing {data} — run the build_*.py scripts first")
        remote_cfg = f"{REMOTE_WORKSPACE}/{stage['config_local']}"
        print(f"[{stage['name']}] pushing {stage['config_local']}")
        run_scp(state, cfg, remote_cfg)
        size_mb = data.stat().st_size // (1024 * 1024)
        print(f"[{stage['name']}] pushing {stage['data_local']} ({size_mb} MB)")
        run_scp(state, data, stage["data_remote"])
    print("\nNext: python3 runpod_train.py train")


def _pipeline_run(state):
    """Run CPT → SFT → DPO with merges between. Idempotent: skip done stages."""
    state["pipeline_started_at"] = state.get("pipeline_started_at", int(time.time()))
    save_state(state)
    for stage in PIPELINE_STAGES:
        _run_one_stage(state, stage)
    print("\nPipeline complete. Final merged model at /workspace/dpo_merged on pod.")
    print("Next: python3 runpod_train.py pull")


def _run_one_stage(state, stage):
    name = stage["name"]
    done_file = f"{REMOTE_WORKSPACE}/{name}.done"
    log_file = f"{REMOTE_WORKSPACE}/{name}.log"
    remote_cfg = f"{REMOTE_WORKSPACE}/{stage['config_local']}"
    output_dir = stage["output_dir"]
    merged_dir = stage["merged_dir"]

    # Skip if previously completed successfully
    r = run_ssh(state, f"cat {done_file} 2>/dev/null", capture=True, check=False)
    if r.stdout.strip() == "OK":
        print(f"[{name}] already complete, skipping")
        return

    # Clean up stale tmux session and marker from any failed run
    run_ssh(state, f"tmux kill-session -t stage_{name} 2>/dev/null || true", check=False)
    run_ssh(state, f"rm -f {done_file}", check=False)

    # Substitute base_model placeholder for SFT/DPO stages
    if stage["needs_base_substitution"]:
        prev = stage["previous_merged"]
        run_ssh(state, f"sed -i 's|__PIPELINE_BASE__|{prev}|g' {remote_cfg}")

    # Build the bash command: train → merge → flatten → cleanup → marker
    rm_prev = f"rm -rf {stage['previous_merged']}; " if stage["previous_merged"] else ""
    bash = (
        "export PATH=/workspace/axolotl-venv/bin:$PATH; "
        "set -o pipefail; "
        f"axolotl train {remote_cfg} 2>&1 | tee {log_file}; "
        "te=${PIPESTATUS[0]}; "
        f"if [ $te -ne 0 ]; then echo \"FAIL train ($te)\" > {done_file}; exit; fi; "
        f"rm -rf {merged_dir} {merged_dir}_parent; "
        f"axolotl merge-lora {remote_cfg} --lora-model-dir={output_dir} --output-dir={merged_dir}_parent 2>&1 | tee -a {log_file}; "
        "me=${PIPESTATUS[0]}; "
        f"if [ $me -ne 0 ]; then echo \"FAIL merge ($me)\" > {done_file}; exit; fi; "
        f"mv {merged_dir}_parent/merged {merged_dir} && rm -rf {merged_dir}_parent; "
        f"if [ ! -f {merged_dir}/config.json ]; then echo \"FAIL flatten (no config.json)\" > {done_file}; exit; fi; "
        f"{rm_prev}"
        f"rm -rf {output_dir}/checkpoint-*; "
        f"echo OK > {done_file}"
    )
    tmux_cmd = f"tmux new-session -d -s stage_{name} {shlex.quote(bash)}"
    print(f"[{name}] launching in tmux (config: {stage['config_local']})")
    run_ssh(state, tmux_cmd)

    # Poll for completion (CPT ~3-4hr, SFT ~2-3hr, DPO ~1hr on a 30B)
    print(f"[{name}] polling every 60s; tail -F {log_file} on pod for live output")
    deadline = time.time() + 8 * 3600
    while time.time() < deadline:
        r = run_ssh(state, f"cat {done_file} 2>/dev/null", capture=True, check=False)
        result = r.stdout.strip()
        if result == "OK":
            print(f"[{name}] done")
            return
        if result.startswith("FAIL"):
            tail = run_ssh(state, f"tail -80 {log_file}", capture=True, check=False)
            print(f"[{name}] FAILED — {result}\n{tail.stdout}")
            sys.exit(1)
        t = run_ssh(state, f"tail -1 {log_file} 2>/dev/null", capture=True, check=False)
        line = t.stdout.strip()
        if line:
            print(f"  [{name}] {line[-200:]}", flush=True)
        time.sleep(60)
    sys.exit(f"[{name}] timeout after 8 hours")


def _pipeline_pull(state):
    """Pull final merged model dir."""
    src_remote = f"{REMOTE_WORKSPACE}/dpo_merged"
    PIPELINE_FINAL_DIR.mkdir(exist_ok=True)
    print(f"Pulling {src_remote}/ -> {PIPELINE_FINAL_DIR}/")
    scp_back(state, f"{src_remote}/.", PIPELINE_FINAL_DIR, recursive=True)
    total = sum(p.stat().st_size for p in PIPELINE_FINAL_DIR.rglob("*") if p.is_file())
    print(f"Pulled {total // (1024*1024)} MB")
    print(f"\nNext: ./convert_to_mlx.sh {PIPELINE_FINAL_DIR.name} crystal-mlx-4bit")


def cmd_merge():
    """Merge the trained LoRA into the base model on the pod (lora mode only)."""
    state = load_state()
    if MODE != "lora":
        print(f"merge is only meaningful in lora mode (current: {MODE})")
        return
    print("Locating most-recent LoRA checkpoint on pod...")
    r = run_ssh(
        state,
        f"ls -td {REMOTE_OUTPUT}/checkpoint-* 2>/dev/null | head -1 || echo {REMOTE_OUTPUT}",
        capture=True,
    )
    adapter_path = r.stdout.strip() or REMOTE_OUTPUT
    print(f"  adapter: {adapter_path}")
    merged_path = f"{REMOTE_WORKSPACE}/merged"
    print(f"Running axolotl merge-lora -> {merged_path}")
    cmd = (
        f"export PATH=/workspace/axolotl-venv/bin:$PATH && "
        f"axolotl merge-lora {REMOTE_CONFIG} "
        f"--lora-model-dir={adapter_path} "
        f"--output-dir={merged_path} 2>&1 | tee {REMOTE_WORKSPACE}/merge.log"
    )
    run_ssh(state, cmd)
    print(f"\nMerged model at {merged_path} on pod. Pull with:")
    print(f"  scp -P {state['ssh_port']} -r root@{state['ssh_host']}:{merged_path} ./runpod-merged")


def cmd_down():
    runpod = setup_runpod()
    state = load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod to terminate.")
        return
    print(f"Terminating pod {pod_id}...")
    runpod.terminate_pod(pod_id)
    cost_estimate = (time.time() - state.get("created_at", time.time())) / 3600 * HOURLY_RATE
    print(f"Approximate cost: ${cost_estimate:.2f}")
    state.pop("pod_id", None)
    state.pop("ssh_host", None)
    state.pop("ssh_port", None)
    save_state(state)


COMMANDS = {
    "up": cmd_up,
    "push": cmd_push,
    "train": cmd_train,
    "tail": cmd_tail,
    "status": cmd_status,
    "pull": cmd_pull,
    "merge": cmd_merge,
    "down": cmd_down,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__.strip())
        print(f"\nCommands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
