"""Show only observed small-subset results and their actual simulator videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SEEDS = (9011, 9012, 9013)


def copy_video(ref: dict, output: Path) -> str:
    source = Path(ref["path"])
    payload = source.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != ref["sha256"] or len(payload) != ref["bytes"]:
        raise ValueError(f"Video differs from its recorded identity: {source}")
    name = checksum[:16] + ".mp4"
    destination = output / name
    if not destination.exists():
        shutil.copyfile(source, destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
        raise ValueError("Published video copy differs")
    return name


def read_condition(path: Path | None, output: Path, *, qf3: bool = False) -> dict:
    if path is None:
        return {"status": "Pending", "episodes": [], "score": None}
    if qf3 and (path / "qf3").is_dir():
        path = path / "qf3"
    worker_path = path / "receipt.json"
    worker = json.loads(worker_path.read_text()) if worker_path.is_file() else {}
    episodes = []
    if qf3 and worker:
        finals = [
            index
            for index, event in enumerate(worker.get("evaluation_events", []))
            if event.get("head") == "learned" and event.get("reason") == "final"
        ]
        stage = f"evaluate-learned-{finals[-1]}" if finals else None
        recordings = {
            row["requested_seed"]: row
            for row in worker.get("subset_video_recordings", [])
            if row["requested_seed"] in SEEDS
            and stage is not None
            and row.get("label", "").startswith(stage + "-")
            and row.get("domain") == "sim"
            and row.get("finalized") is True
        }
        for pin in worker.get("rollouts", []):
            raw = Path(pin["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
                raise ValueError("QF3 rollout differs from its receipt")
            record = json.loads(raw)
            if stage is None or record.get("domain") != "sim" or record.get("stage") != stage:
                continue
            rollout = record.get("rollout", {})
            for row in rollout.get("worlds", []):
                seed = row.get("requested_seed")
                if seed not in SEEDS or row.get("completed") is not True:
                    continue
                if row["actual_seed"] != seed:
                    raise ValueError("QF3 actual layout differs from the completed baseline layout")
                recording = recordings.get(seed, {})
                if recording.get("actual_seed") != row["actual_seed"]:
                    raise ValueError("QF3 evaluation video and rollout layout differ")
                video = recording.get("video")
                if not video:
                    raise ValueError("Completed QF3 evaluation is missing its actual video")
                episodes.append(
                    {
                        "seed": seed,
                        "actual_seed": row["actual_seed"],
                        "steps": row["steps"],
                        "success": row["paper_success"],
                        "native_success": row["native_success"],
                        "bottles_ever": sum(row["tracker"]["ever_placed"])
                        if "tracker" in row else None,
                        "video": copy_video(video, output),
                    }
                )
    else:
        host = path / "host" if (path / "host").is_dir() else path
        for episode_path in sorted(host.glob("episode-*.json")):
            row = json.loads(episode_path.read_text())
            seed = row["requested_seed"]
            if seed in SEEDS and row["completed"] is True:
                if row["seed"] != seed:
                    raise ValueError("Host actual layout differs from the completed baseline layout")
                episodes.append(
                    {
                        "seed": seed,
                        "actual_seed": row["seed"],
                        "steps": row["steps"],
                        "success": row["task_success"],
                        "native_success": row["native_current_success"],
                        "bottles_ever": sum(row["ever_placed"])
                        if "ever_placed" in row else None,
                        "video": copy_video(row["video"], output),
                    }
                )
    if len({row["seed"] for row in episodes}) != len(episodes):
        raise ValueError("Duplicate matched evaluation layouts")
    complete = (
        {row["seed"] for row in episodes} == set(SEEDS)
        and worker.get("status") == "completed"
    )
    state = "Complete" if complete else f"Running: {len(episodes)}/3 episodes complete"
    if worker.get("status") in {"failed", "budget_stopped", "early_stopped"}:
        state = f"{worker['status']}: {len(episodes)}/3 episodes complete"
    return {
        "status": state,
        "episodes": episodes,
        "score": f"{sum(row['success'] for row in episodes)}/3" if complete else None,
        "source": str(path),
        "worker_status": worker.get("status"),
    }


HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Small original ABC-VLA comparison</title><style>
body{background:#101720;color:#e8edf3;font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 22px}h1{font-size:29px}p{max-width:980px;color:#c0ccd9;line-height:1.5}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.card{background:#1a2532;border:1px solid #3b4b60;border-radius:12px;padding:18px}video{width:100%;background:#080c12}.score{font-size:30px;margin:12px 0;color:#94dcbb}.note{font-size:14px;color:#b2bdcb}select,button{background:#233348;color:white;border:1px solid #677a93;padding:9px;border-radius:6px;margin-right:8px}a{color:#9fc8ff}.table-scroll{overflow-x:auto}table{min-width:650px;border-collapse:collapse;width:100%;margin:24px 0}td,th{text-align:left;padding:10px;border-bottom:1px solid #34455a}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style><h1>ABC-VLA · small simulator comparison</h1>
<p>Same original ABC-VLA checkpoint and MJWarp simulator. Five bottles, three shared layouts, five flow steps and fifteen executed controls per plan. Each episode has at most 1,000 controls. Clips show actual simulator observations.</p>
<p class="note">The score counts whether all five bottles entered the bin at least once. The original current-success judge is shown separately. This is a short transfer diagnostic. Three layouts and one training seed cannot establish a reliable gain or reproduce paper percentages.</p>
<label>Layout <select id="seed"><option>9011</option><option>9012</option><option>9013</option></select></label><button id="play">Play all clips</button><button onclick="location.reload()">Refresh results</button><div class="grid" id="cards"></div>
<div class="table-scroll"><table><thead><tr><th>Condition</th><th>Training</th><th>Implementation</th></tr></thead><tbody><tr><td>Frozen ABC-VLA</td><td>No post-training</td><td>Original ABC policy and simulator</td></tr><tr><td>ABC-VLA + ResFiT</td><td>256 stored-row warmup, 32 trained controls; 160 critic / 32 actor updates; 716 genuine successful demo transitions</td><td>Unchanged author learner. Explicit ABC I/O and short-budget adapter. Original action scaling can alter the base even with zero residual.</td></tr><tr><td>ABC-VLA + QF3</td><td>At most two 1,000-control training episodes; 128 critic / 16 actor updates</td><td>Paper actor equations and rank4 head LoRA. Author code and exact initializer unavailable in the checked public sources; several critic settings are local choices.</td></tr></tbody></table></div>
<p class="note">Training budgets and data differ across methods. Scores are descriptive observations under shared evaluation conditions. Baseline defaults were configured to QF3's deployment sampling for this comparison.</p>
<script>const data=__DATA__;const names=['Frozen ABC-VLA','ABC-VLA + original ResFiT','ABC-VLA + QF3'];function draw(){const seed=Number(document.querySelector('#seed').value);document.querySelector('#cards').replaceChildren();data.forEach((row,i)=>{const card=document.createElement('section');card.className='card';const heading=document.createElement('h2');heading.textContent=names[i];card.append(heading);const status=document.createElement('div');status.textContent=row.status;card.append(status);const score=document.createElement('div');score.className='score';score.textContent=row.score===null?'Score pending':row.score+' successes';card.append(score);const ep=row.episodes.find(x=>x.seed===seed);if(ep&&ep.video){const video=document.createElement('video');video.controls=true;video.preload='metadata';video.src=ep.video;card.append(video)}const detail=document.createElement('p');detail.className='note';detail.textContent=ep?('Layout '+ep.seed+' · '+ep.steps+' controls'+(ep.bottles_ever===null?'':' · '+ep.bottles_ever+'/5 bottles entered')+' · shared success '+ep.success+' · original current success '+ep.native_success+(ep.actual_seed!==ep.seed?' · actual resampled seed '+ep.actual_seed:'')):'No completed rollout for this layout yet.';card.append(detail);document.querySelector('#cards').append(card)})}document.querySelector('#seed').onchange=draw;document.querySelector('#play').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})});draw();</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--resfit", type=Path)
    parser.add_argument("--qf3", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = [
        read_condition(args.baseline, args.output),
        read_condition(args.resfit, args.output),
        read_condition(args.qf3, args.output, qf3=True),
    ]
    (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    html = HTML.replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))
    temporary = args.output / "index.html.tmp"
    temporary.write_text(html)
    temporary.replace(args.output / "index.html")
    print(json.dumps([{key: row[key] for key in ("status", "score")} for row in data]))


if __name__ == "__main__":
    main()
