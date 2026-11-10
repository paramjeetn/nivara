#!/usr/bin/env python3
"""Nivara Dual-Screen Demo Launcher.

Starts both:
1. Primary Product Dashboard:   http://localhost:8501 (src/campus_energy/dashboard/app.py)
2. Presenter Controller Sidecar: http://localhost:8502 (src/campus_energy/presenter/controller.py)
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from campus_energy.storage.seed import seed_operational_history


def main() -> int:
    print("=" * 65)
    print("⚡ Nivara Dual-Display Demonstration Launcher")
    print("=" * 65)

    # 1. Initialize clean lived-in baseline before starting servers
    print("📦 Seeding pristine operational history (3 verified WOs, 5 alerts)...")
    try:
        seed_operational_history()
        print("✅ Operational baseline seeded successfully.")
    except Exception as exc:
        print(f"⚠️  Note: Operational seed encountered: {exc}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Commands for both Streamlit processes
    cmd_app = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(REPO_ROOT / "src" / "campus_energy" / "dashboard" / "app.py"),
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]

    cmd_controller = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(REPO_ROOT / "src" / "campus_energy" / "presenter" / "controller.py"),
        "--server.port",
        "8502",
        "--server.headless",
        "true",
    ]

    procs: list[subprocess.Popen] = []

    try:
        print("\n🚀 Starting Primary Product on Port 8501...")
        p_app = subprocess.Popen(cmd_app, cwd=str(REPO_ROOT), env=env)
        procs.append(p_app)

        # Brief pause to let port 8501 bind
        time.sleep(1.0)

        print("🎙️ Starting Presenter Controller Sidecar on Port 8502...")
        p_controller = subprocess.Popen(cmd_controller, cwd=str(REPO_ROOT), env=env)
        procs.append(p_controller)

        print("\n" + "=" * 65)
        print("🌟 Both Demonstration Services Active:")
        print("   🖥️  Product Dashboard:       http://localhost:8501")
        print("   🎙️  Presenter Sidecar:      http://localhost:8502")
        print("=" * 65)
        print("💡 Open Port 8501 on primary display and Port 8502 on companion monitor.")
        print("⌨️  Press Ctrl+C to terminate both servers cleanly.\n")

        # Keep parent alive until interrupted or a child exits
        while True:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    print(f"Subprocess {p.args} exited with code {ret}")
                    return ret
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down demo servers gracefully...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("👋 All demonstration processes stopped cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
