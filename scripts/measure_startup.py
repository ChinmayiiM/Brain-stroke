"""Measure real `streamlit run` start-up: when does the UI first appear in the browser?

Usage (project root, venv active):
    python scripts/measure_startup.py

Starts `streamlit run app/streamlit_app.py` headless on a test port, connects like a
browser (websocket), requests the first script run and records, from the moment the
command is launched:
    server ready (health 'ok') · first UI element received · header received · script finished
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
PORT = 8595


async def first_run(t0: float) -> dict[str, float]:
    marks: dict[str, float] = {}
    ws = await connect(f"ws://localhost:{PORT}/_stcore/stream", subprotocols=["streamlit"], max_size=None)
    back = BackMsg()
    back.rerun_script.query_string = ""
    await ws.send(back.SerializeToString())
    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=300)
            if isinstance(raw, str):
                continue
            msg = ForwardMsg()
            msg.ParseFromString(raw)
            kind = msg.WhichOneof("type")
            now = time.perf_counter() - t0
            if kind == "delta":
                marks.setdefault("first UI element", now)
                if "Brain Stroke Detection" in str(msg.delta):
                    marks.setdefault("header visible", now)
                if "Upload Brain MRI Image" in str(msg.delta):
                    marks.setdefault("upload widget visible", now)
            elif kind == "script_finished":
                marks["first script run finished"] = now
                return marks
    finally:
        await ws.close()


def main() -> int:
    if (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")
    t0 = time.perf_counter()
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                               "--server.headless", "true", "--server.port", str(PORT)], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while time.perf_counter() - t0 < 180:
            try:
                if urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2).read() == b"ok":
                    break
            except OSError:
                time.sleep(0.2)
        ready = time.perf_counter() - t0
        marks = asyncio.run(first_run(t0))
    finally:
        server.terminate()
        server.wait(timeout=30)
    print(f"{'server ready (health ok)':<28} {ready:6.1f} s after launch")
    for name, value in marks.items():
        print(f"{name:<28} {value:6.1f} s after launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
