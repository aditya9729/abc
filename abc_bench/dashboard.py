"""Read-only, localhost benchmark evidence dashboard. No training is launched."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

DEFAULT_EMBODIMENTS = [
    {"id": "native_yam", "label": "Native YAM", "status": "unavailable"},
    {"id": "r1lite", "label": "R1 Lite", "status": "unavailable"},
]
HTML = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ABC · Benchmark evidence</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10161e;color:#e8eef6}body{max-width:1500px;margin:auto;padding:32px}h1{font-size:34px;margin-bottom:8px}p{color:#aab9ca;line-height:1.5}.tag{color:#70deba;letter-spacing:2px;font-size:12px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:26px 0}.card,.diagram{background:#192330;border:1px solid #2d3d50;border-radius:12px;padding:20px}.card h2{margin:0 0 12px;font-size:20px}.status{padding:4px 8px;border-radius:5px;background:#344253;font-size:12px}.completed,.passed{background:#145442}.blocked,.failed{background:#6a343a}.toolbar{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}select,button{background:#1c2a39;border:1px solid #43566b;color:#e8eef6;padding:10px;border-radius:6px}.scroll{overflow:auto;border:1px solid #2d3d50;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:14px;text-align:left;border-bottom:1px solid #2d3d50;vertical-align:top}th{color:#9caec2;white-space:nowrap}td small{display:block;color:#9caec2;margin-top:5px;max-width:230px}a{color:#7fcaff}details{margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#bfd0e2;font-size:12px}.diagram svg{width:100%;height:auto}footer{font-size:12px;color:#9caec2;margin-top:24px}#error{color:#ffabb1}#empty{padding:28px;color:#aab9ca}label{font-size:13px;color:#b4c3d5}@media(max-width:800px){body{padding:16px}.cards{grid-template-columns:1fr}}
</style><div class="tag">SADHANA / READ-ONLY EXPERIMENT VIEW</div><h1>Benchmark evidence</h1><p>Compare recorded experiments. A smoke test checks execution. A benchmark measures task performance.<br>Unavailable values are not measured. Diagnostic implementations do not establish paper reproduction.</p><div id="error" role="alert"></div><div class="cards" id="embodiments"></div>
<div class="toolbar"><label>Method <select id="algorithm"><option value="">All methods</option></select></label><label>Embodiment <select id="embodiment"><option value="">All embodiments</option></select></label><label>Evidence <select id="phase"><option value="">All phases</option><option>smoke</option><option>benchmark</option></select></label><label>Status <select id="status"><option value="">All statuses</option></select></label><button id="refresh">Refresh evidence</button></div>
<div class="scroll"><table><thead><tr><th>Method / scope</th><th>Embodiment / task</th><th>Run / seed</th><th>Success</th><th>Latency p50 / p95</th><th>Reward / completion</th><th>Budget / execution</th><th>Status / evidence</th></tr></thead><tbody id="runs"></tbody></table><div id="empty" hidden>No recorded runs match these filters.</div></div>
<h2>Architecture and team</h2><div class="diagram"><svg viewBox="0 0 1000 165" role="img" aria-label="Runner writes measured receipts. Read-only dashboard presents receipts and approved artifacts. Independent reviewer checks the evidence."><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#70deba"/></marker></defs><g fill="#25364a" stroke="#597086"><rect x="10" y="30" width="210" height="80" rx="10"/><rect x="280" y="30" width="210" height="80" rx="10"/><rect x="550" y="30" width="210" height="80" rx="10"/><rect x="810" y="30" width="180" height="80" rx="10"/></g><g fill="#e8eef6" text-anchor="middle" font-family="system-ui" font-size="15"><text x="115" y="62">Gym + method runner</text><text x="115" y="85">Coordinator / algorithm team</text><text x="385" y="62">Measured run receipts</text><text x="385" y="85">Metrics + artifact paths</text><text x="655" y="62">Read-only dashboard</text><text x="655" y="85">UI agent</text><text x="900" y="62">Evidence review</text><text x="900" y="85">Independent reviewer</text></g><g stroke="#70deba" stroke-width="2" marker-end="url(#arrow)"><path d="M220 70 H275"/><path d="M490 70 H545"/><path d="M760 70 H805"/></g><text x="500" y="145" text-anchor="middle" fill="#aab9ca" font-size="13">Data flow → No dashboard control flow to training or robot hardware.</text></svg></div><footer id="updated"></footer>
<script>
let data={runs:[],embodiments:[]};const $=id=>document.getElementById(id);const present=v=>v!==null&&v!==undefined;const number=(v,suffix='')=>present(v)?(typeof v==='number'&&!Number.isInteger(v)?v.toFixed(2):String(v))+suffix:'Unavailable';function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}function badge(v){return el('span',v||'unavailable','status '+(['completed','passed','blocked','failed'].includes(v)?v:''))}function cell(row,text,sub){const c=el('td',text);if(sub)c.append(el('small',sub));row.append(c);return c}function options(id,key){const current=$(id).value;$(id).replaceChildren(el('option','All '+id+'s'));$(id).firstChild.value='';[...new Set(data.runs.map(r=>r[key]).filter(Boolean))].sort().forEach(v=>{const o=el('option',v);o.value=v;$(id).append(o)});$(id).value=current}
function render(){const filtered=data.runs.filter(r=>['algorithm','embodiment','phase','status'].every(k=>!$(k).value||r[k]===$(k).value));$('runs').replaceChildren();$('empty').hidden=filtered.length>0;for(const r of filtered){const row=el('tr'),m=r.metrics||{},lat=m.latency_ms||{},b=r.budget||{};cell(row,r.algorithm||'Unavailable',r.method_fidelity||r.claim_scope||'Scope unavailable');cell(row,r.embodiment||'Unavailable',r.task);cell(row,r.run_id||'Unavailable',(r.phase||'phase unavailable')+' · seed '+number(r.seed));cell(row,present(m.successes)&&present(m.episodes)?`${m.successes} / ${m.episodes}`:'Unavailable');cell(row,number(lat.p50,' ms')+' / '+number(lat.p95,' ms'),m.latency_scope);cell(row,number(m.reward),number(m.completion_time_seconds,' s'));cell(row,number(b.wall_seconds,' s budget'),number(m.elapsed_seconds,' s elapsed')+' · '+number(m.simulation_steps??b.steps,' steps'));const c=cell(row);c.append(badge(r.status));for(const blocker of r.blockers||[])c.append(el('small',blocker));for(const a of r.artifacts||[]){if(!a.url)continue;const link=el('a',a.label||a.path);link.href=a.url;link.target='_blank';link.rel='noopener';c.append(el('br'),link)}if(r.checkpoint_path)c.append(el('small','Checkpoint: '+r.checkpoint_path));const details=el('details');details.append(el('summary','Run receipt'),el('pre',JSON.stringify(r,null,2)));c.append(details);$('runs').append(row)}}
async function load(){try{if(window.BENCHMARK_SNAPSHOT){data=window.BENCHMARK_SNAPSHOT}else{const response=await fetch('/api/results',{cache:'no-store'});if(!response.ok)throw Error('Evidence request failed: '+response.status);data=await response.json()}$('error').textContent=(data.errors||[]).join(' · ');$('embodiments').replaceChildren();for(const e of data.embodiments){const card=el('div',undefined,'card');card.append(el('h2',e.label||e.id),badge(e.status));for(const b of e.blockers||[])card.append(el('p',b));$('embodiments').append(card)}['algorithm','embodiment','status'].forEach(k=>options(k,k));render();$('updated').textContent='Read at '+new Date().toLocaleString()+'. Receipts remain the source of evidence. STE guidance used; full ASD-STE100 compliance was not checked.'}catch(e){$('error').textContent=String(e)}}['algorithm','embodiment','phase','status'].forEach(k=>$(k).onchange=render);$('refresh').onclick=load;load();
</script></html>"""


def reject_nonfinite(value: str) -> None:
    """JSON receipts cannot contain NaN or infinite measured values."""
    raise ValueError(f"Non-finite JSON value: {value}")


def artifact_path(root: Path, raw: str) -> Path | None:
    """Accept existing receipt artifacts only inside the resolved results root."""
    candidate = (root / raw).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        return None
    return candidate


def load_results(root: Path) -> tuple[dict, dict[str, Path]]:
    """Load canonical receipts; expose only explicitly named safe artifacts."""
    result: dict = {
        "schema_version": 1,
        "runs": [],
        "embodiments": DEFAULT_EMBODIMENTS,
        "errors": [],
    }
    paths = [root / "results.json"]
    if not paths[0].exists():
        paths = sorted((root / "runs").glob("*.json"))
    for path in paths:
        if not path.exists():
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            result["errors"].append(
                f"Rejected receipt outside results root: {path.name}"
            )
            continue
        try:
            payload = json.loads(path.read_text(), parse_constant=reject_nonfinite)
            if not isinstance(payload, dict):
                raise TypeError("receipt must be an object")
            runs = payload.get("runs", [payload] if "run_id" in payload else [])
            if not isinstance(runs, list) or any(not isinstance(r, dict) for r in runs):
                raise ValueError("runs must be a list of objects")
            result["runs"].extend(runs)
            if "embodiments" in payload:
                embodiments = payload["embodiments"]
                if not isinstance(embodiments, list) or any(
                    not isinstance(e, dict) for e in embodiments
                ):
                    raise ValueError("embodiments must be a list of objects")
                result["embodiments"] = embodiments
        except (OSError, ValueError, TypeError) as exc:
            result["errors"].append(f"Cannot read {path.name}: {exc}")
    artifacts: dict[str, Path] = {}
    for run in result["runs"]:
        supplied = run.get("artifacts", [])
        if not isinstance(supplied, list):
            result["errors"].append("Ignored invalid artifacts list")
            supplied = []
        safe = []
        for item in supplied:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            path = artifact_path(root, item["path"])
            clean = {"label": str(item.get("label", "Artifact")), "path": item["path"]}
            if path is not None:
                token = hashlib.sha256(
                    str(path.relative_to(root.resolve())).encode()
                ).hexdigest()[:24]
                artifacts[token] = path
                clean["url"] = f"/artifacts/{token}"
            safe.append(clean)
        run["artifacts"] = safe
    return result, artifacts


def make_server(
    root: Path, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    """Create a read-only server. Call server_close after shutdown."""
    root = root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = unquote(urlsplit(self.path).path)
            if route == "/":
                self.respond(200, HTML.encode(), "text/html; charset=utf-8")
            elif route == "/api/results":
                result, _ = load_results(root)
                self.respond(
                    200,
                    json.dumps(result, allow_nan=False).encode(),
                    "application/json",
                )
            elif route.startswith("/artifacts/"):
                _, paths = load_results(root)
                path = paths.get(route.removeprefix("/artifacts/"))
                # Recheck symlinks at request time; no arbitrary file browsing.
                if path is None or artifact_path(root, str(path)) is None:
                    self.respond(404, b"Artifact not available", "text/plain")
                    return
                try:
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        mimetypes.guess_type(path.name)[0]
                        or "application/octet-stream",
                    )
                    self.send_header("Content-Length", str(path.stat().st_size))
                    self.send_header(
                        "Content-Disposition",
                        "inline"
                        if path.suffix.lower() in {".mp4", ".webm", ".png", ".jpg"}
                        else "attachment",
                    )
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            self.wfile.write(chunk)
                except (OSError, BrokenPipeError):
                    return
            else:
                self.respond(404, b"Not found", "text/plain")

        def respond(self, status: int, content: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            return  # Do not log receipt contents or arbitrary user query strings.

    return ThreadingHTTPServer((host, port), Handler)


def export_snapshot(root: Path, destination: Path) -> None:
    """Write a portable interactive view of the current verified receipt data."""
    data, _ = load_results(root)
    for run in data["runs"]:
        for artifact in run.get("artifacts", []):
            if artifact.get("url"):
                source = artifact_path(root, artifact["path"])
                if source is None:
                    artifact.pop("url", None)
                    continue
                artifact["url"] = quote(
                    os.path.relpath(source, destination.parent.resolve())
                )
    encoded = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    document = HTML.replace(
        "<script>", "<script>window.BENCHMARK_SNAPSHOT=" + encoded + ";", 1
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("outputs/bench"))
    parser.add_argument(
        "--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1"
    )
    parser.add_argument(
        "--export", type=Path, help="Write an offline interactive snapshot and exit"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host == "::1":
        parser.error("Use 127.0.0.1 or localhost; this server uses IPv4.")
    if args.export is not None:
        export_snapshot(args.results_root, args.export)
        return
    server = make_server(args.results_root, args.host, args.port)
    print(f"Benchmark evidence: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
