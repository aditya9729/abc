"""Root-only CPU browser check after both actual recordings are published."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

BROWSER = "/home/user/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell"
KEYS = ("robot_log_replay", "reconstructed_laundry_scene")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Root's existing localhost viewer URL")
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    url = urlsplit(args.url)
    require(
        url.scheme == "http"
        and url.hostname == "127.0.0.1"
        and url.port is not None
        and url.username is None
        and url.password is None
        and not url.query
        and not url.fragment,
        "Use the root-authorized HTTP127.0.0.1 viewer URL",
    )
    require(args.url.endswith("/"), "Viewer URL must end in /")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for this CPU browser check")
    require(
        args.media.is_absolute()
        and args.media.resolve().is_relative_to("/home/user/aditya")
        and args.media.is_file()
        and not args.media.is_symlink(),
        "Use the actual local media manifest",
    )
    require(
        args.output_dir.is_absolute()
        and args.output_dir.resolve().is_relative_to("/home/user/aditya")
        and not args.output_dir.exists(),
        "Use a fresh Aditya output directory",
    )
    data = args.media.read_bytes()
    manifest = json.loads(data)
    require(
        manifest["schema"] == "taku.local-scene-viewer/v1"
        and all(manifest[key]["status"] == "accepted_complete" for key in KEYS),
        "Both actual recordings must be accepted before this browser check",
    )
    with urlopen(args.url + "media.json", timeout=15) as response:
        require(
            response.status == 200 and response.read(len(data) + 1) == data,
            "Served manifest differs",
        )
    args.output_dir.mkdir(parents=True)
    temporary = tempfile.mkdtemp(prefix="taku-browser-", dir="/home/user/aditya/t")
    os.environ["TMPDIR"] = temporary
    tempfile.tempdir = temporary
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=BROWSER,
            headless=True,
            timeout=30000,
            args=[
                "--disable-gpu",
                "--disable-gpu-compositing",
                "--use-gl=disabled",
                "--disable-webgl",
            ],
        )
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 1000})
            page.set_default_timeout(15000)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.url, wait_until="domcontentloaded", timeout=20000)
            page.locator("#reconstructed_laundry_scene-player video").wait_for(state="visible")
            require(page.locator("#error").inner_text() == "", "Viewer reported an error")
            page.locator(".player video").evaluate_all("vs=>vs.forEach(v=>v.load())")
            page.wait_for_function(
                "Array.from(document.querySelectorAll('.player video')).length===2 && Array.from(document.querySelectorAll('.player video')).every(v=>v.readyState>=2 && Number.isFinite(v.duration))"
            )
            actual = []
            for key in KEYS:
                clip = manifest[key]
                video = page.locator("#" + key + "-player video")
                metadata = video.evaluate(
                    "v=>({src:v.getAttribute('src'),duration:v.duration,width:v.videoWidth,height:v.videoHeight,controls:v.controls})"
                )
                require(
                    metadata["src"] == clip["video"]
                    and metadata["controls"]
                    and metadata["width"] == clip["width"]
                    and metadata["height"] == clip["height"]
                    and abs(metadata["duration"] - clip["frames"] / clip["fps"]) < 0.02,
                    "Native recording metadata differs",
                )
                video.evaluate("v=>{v.currentTime=v.duration*.75}")
                page.wait_for_function(
                    "s=>{const v=document.querySelector(s);return !v.seeking && v.readyState>=2 && Math.abs(v.currentTime-v.duration*.75)<.03}",
                    arg="#" + key + "-player video",
                )
                video.evaluate(
                    "v=>{v.muted=true;return Promise.race([v.play(),new Promise((_,reject)=>setTimeout(()=>reject(new Error('Play exceeded8s')),8000))])}"
                )
                page.wait_for_function(
                    "s=>{const v=document.querySelector(s);return !v.paused && v.currentTime>v.duration*.75+.1}",
                    arg="#" + key + "-player video",
                )
                video.evaluate("v=>v.pause()")
                actual.append({"key": key, "metadata": metadata, "seek_and_play": True})
            page.evaluate(
                "window.savedTakuVideos=Array.from(document.querySelectorAll('.player video'))"
            )
            positions = page.locator(".player video").evaluate_all("vs=>vs.map(v=>v.currentTime)")
            page.evaluate(
                "()=>Promise.race([refreshMedia(),new Promise((_,reject)=>setTimeout(()=>reject(new Error('Refresh exceeded10s')),10000))])"
            )
            require(page.locator("#error").inner_text() == "", "Media refresh reported an error")
            require(
                page.evaluate(
                    "positions=>savedTakuVideos.every((v,i)=>v===document.querySelectorAll('.player video')[i] && Math.abs(v.currentTime-positions[i])<.03)",
                    positions,
                ),
                "Refresh replaced or reset players",
            )
            page.wait_for_function(
                "document.querySelector('.reference img').complete && document.querySelector('.reference img').naturalWidth>0"
            )
            require(
                page.evaluate("document.documentElement.scrollWidth<=innerWidth+1"),
                "Desktop horizontal overflow",
            )
            page.screenshot(path=str(args.output_dir / "desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            require(
                page.evaluate("document.documentElement.scrollWidth<=innerWidth+1"),
                "Mobile horizontal overflow",
            )
            page.screenshot(path=str(args.output_dir / "mobile.png"), full_page=True)
            require(not errors, "Browser page errors: " + str(errors))
            require(args.media.read_bytes() == data, "Media manifest changed during browser check")
            result = {
                "status": "passed",
                "scope": "Actual local visual-replay page; CPU browser only",
                "url": args.url,
                "media_manifest": {
                    "path": str(args.media),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
                "videos": actual,
                "seek_play_refresh": True,
                "desktop_mobile_no_overflow": True,
                "browser_version": browser.version,
                "gpu_disabled": True,
                "page_errors": errors,
                "limits": "Display/media evidence, not autonomous control, dynamic cloth, task success or exact room reproduction.",
            }
            with (args.output_dir / "BROWSER_RESULT.json").open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(json.dumps(result))
        finally:
            browser.close()


if __name__ == "__main__":
    main()
