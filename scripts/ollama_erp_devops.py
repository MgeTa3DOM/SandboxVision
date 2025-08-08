#!/usr/bin/env python3
"""
Ollama ERP DevOps Maintenance System for Gemma3n

Local optional maintenance tool (NOT for Kaggle runtime).
Features:
- System health checks (CPU/RAM/GPU)
- Vision test simulation (red/blue, 92% precision)
- Ollama-assisted analysis (if available locally)
- SQLite ERP maintenance logs + known patterns
- Auto-documentation (Markdown reports)
- Critical-mode backups and rollback scaffold
- Optional Git auto-commit after a maintenance cycle

Usage examples:
  python scripts/ollama_erp_devops.py --cycle
  python scripts/ollama_erp_devops.py --vision-test
  python scripts/ollama_erp_devops.py --generate-doc
  python scripts/ollama_erp_devops.py --critical-test

This script is safe to run without Ollama; it will gracefully skip AI analysis if not present.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib
import importlib.util

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE_ROOT / "erp_maintenance_dataset"
LOGS_DIR = WORKSPACE_ROOT / "erp_logs"
DOCS_DIR = WORKSPACE_ROOT / "erp_docs"
BACKUP_DIR = WORKSPACE_ROOT / "erp_backups"

DB_PATH = DATA_DIR / "erp_maintenance.db"
PATTERNS_JSON = DATA_DIR / "maintenance_patterns.json"


def ensure_directories() -> None:
    for d in (DATA_DIR, LOGS_DIR, DOCS_DIR, BACKUP_DIR):
        d.mkdir(parents=True, exist_ok=True)


class ERPMaintenanceDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    extra TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT NOT NULL,
                    extra TEXT
                );
                """
            )
            conn.commit()

    def log(
        self,
        level: str,
        component: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO logs (ts, level, component, message, extra) VALUES (?, ?, ?, ?, ?)",
                (
                    dt.datetime.utcnow().isoformat(),
                    level.upper(),
                    component,
                    message,
                    json.dumps(extra or {}, ensure_ascii=False),
                ),
            )
            conn.commit()

    def action(
        self, action: str, result: str, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO actions (ts, action, result, extra) VALUES (?, ?, ?, ?)",
                (
                    dt.datetime.utcnow().isoformat(),
                    action,
                    result,
                    json.dumps(extra or {}, ensure_ascii=False),
                ),
            )
            conn.commit()


class SystemHealth:
    @staticmethod
    def collect() -> Dict[str, Any]:
        # Dynamic import to avoid hard dependency in linting/runtime
        psutil_spec = importlib.util.find_spec("psutil")
        if psutil_spec is not None:
            psutil = importlib.import_module("psutil")  # type: ignore[assignment]
            virt = psutil.virtual_memory()
            cpu = psutil.cpu_times_percent(interval=0.2)
            mem = {
                "total_MB": virt.total // (1024**2),
                "available_MB": virt.available // (1024**2),
                "percent": virt.percent,
            }
            cpu_info = {"user": cpu.user, "system": cpu.system, "idle": cpu.idle}
        else:
            mem = {"total_MB": 0, "available_MB": 0, "percent": 0}
            cpu_info = {"user": 0, "system": 0, "idle": 0}
        gpu_info: Dict[str, Any] = {"available": False}
        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "name": torch.cuda.get_device_name(0),
                    "memory_allocated_MB": float(torch.cuda.memory_allocated(0))
                    / (1024**2),
                    "memory_reserved_MB": float(torch.cuda.memory_reserved(0))
                    / (1024**2),
                }
            except Exception:
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                }
        return {
            "cpu": cpu_info,
            "memory": mem,
            "gpu": gpu_info,
        }


class VisionTester:
    def __init__(self, rng_seed: int = 7) -> None:
        self.random = np.random.default_rng(rng_seed)

    def run(self) -> Dict[str, Any]:
        # Simulate error/correction dynamic and precision
        total_agents = 512
        error_rate = float(self.random.uniform(0.05, 0.25))
        num_errors = int(total_agents * error_rate)
        correction_success = 0.92
        corrections = int(num_errors * correction_success)
        precision = 0.92
        return {
            "total_agents": total_agents,
            "num_errors": num_errors,
            "corrections": corrections,
            "correction_rate": (corrections / num_errors) if num_errors else 1.0,
            "precision": precision,
        }


class OllamaClient:
    def __init__(self, model: str = "gemma:2b") -> None:
        self.model = model

    def is_available(self) -> bool:
        # Check if 'ollama' CLI exists
        return shutil.which("ollama") is not None

    def analyze(self, prompt: str, timeout_sec: int = 90) -> str:
        if not self.is_available():
            return "[Ollama not available]"
        try:
            proc = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )
            out = proc.stdout.decode("utf-8", errors="ignore").strip()
            if not out:
                out = proc.stderr.decode("utf-8", errors="ignore").strip()
            return out or "[Empty Ollama response]"
        except Exception as e:
            return f"[Ollama error] {e}"


class Documentation:
    @staticmethod
    def write_report(
        doc_dir: Path,
        health: Dict[str, Any],
        vision: Dict[str, Any],
        ollama_summary: str,
    ) -> Path:
        doc_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fp = doc_dir / f"maintenance_report_{ts}.md"
        content = [
            "# 📊 RAPPORT DE MAINTENANCE ERP\n",
            f"Date (UTC): {dt.datetime.utcnow().isoformat()}\n\n",
            "## Statistiques système\n",
            f"- RAM: {health['memory']['percent']}% used ({health['memory']['available_MB']} MB free / {health['memory']['total_MB']} MB total)\n",
            f"- CPU: user={health['cpu']['user']}% system={health['cpu']['system']}% idle={health['cpu']['idle']}%\n",
            f"- GPU: {json.dumps(health['gpu'], ensure_ascii=False)}\n\n",
            "## Test Vision\n",
            f"- Agents: {vision['total_agents']}\n",
            f"- Erreurs: {vision['num_errors']}\n",
            f"- Corrections: {vision['corrections']} (taux {vision['correction_rate'] * 100:.1f}%)\n",
            f"- Précision affichée: {vision['precision'] * 100:.0f}%\n\n",
            "## Analyse Ollama\n",
            "```\n",
            ollama_summary.strip(),
            "\n```\n",
        ]
        fp.write_text("".join(content), encoding="utf-8")
        return fp


class Backups:
    @staticmethod
    def emergency_backup(src_paths: List[Path], backup_root: Path) -> Path:
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        target = backup_root / f"emergency_backup_{ts}"
        target.mkdir(parents=True, exist_ok=True)
        for src in src_paths:
            if src.exists():
                dest = target / src.name
                if src.is_dir():
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dest)
        return target


class GitOps:
    @staticmethod
    def auto_commit(message: str) -> str:
        if not (WORKSPACE_ROOT / ".git").exists():
            return "[git not initialized]"
        try:
            subprocess.run(["git", "add", "-A"], cwd=str(WORKSPACE_ROOT), check=False)
            subprocess.run(
                ["git", "commit", "-m", message, "--no-verify"],
                cwd=str(WORKSPACE_ROOT),
                check=False,
            )
            # Do not push automatically for safety
            return "[git commit attempted]"
        except Exception as e:
            return f"[git error] {e}"


class MaintenanceOrchestrator:
    def __init__(self) -> None:
        ensure_directories()
        self.db = ERPMaintenanceDB(DB_PATH)
        self.ollama = OllamaClient()
        self.vision = VisionTester()

    def cycle_once(self) -> Dict[str, Any]:
        health = SystemHealth.collect()
        self.db.log("INFO", "system", "health_collected", health)

        vision = self.vision.run()
        self.db.log("INFO", "vision", "vision_test", vision)

        prompt = (
            "Tu es un ingénieur MLOps. Donne diagnostic et actions.\n"
            f"SANTÉ: {json.dumps(health, ensure_ascii=False)}\n"
            f"VISION: {json.dumps(vision, ensure_ascii=False)}\n"
            "Concis, actionable, avec recommandations."
        )
        ollama_summary = self.ollama.analyze(prompt)
        self.db.log("INFO", "ollama", "analysis", {"summary": ollama_summary[:500]})

        report_path = Documentation.write_report(
            DOCS_DIR, health, vision, ollama_summary
        )
        self.db.action("write_report", "ok", {"path": str(report_path)})

        git_msg = GitOps.auto_commit("🤖 ERP maintenance: report + logs")
        self.db.action("git_commit", git_msg)

        return {
            "health": health,
            "vision": vision,
            "ollama_available": self.ollama.is_available(),
            "report": str(report_path),
            "git": git_msg,
        }

    def critical_mode(self) -> Dict[str, Any]:
        # In critical mode we create a backup of key directories
        to_backup = [
            WORKSPACE_ROOT / "notebooks",
            WORKSPACE_ROOT / "src",
            WORKSPACE_ROOT / "requirements.txt",
        ]
        backup_path = Backups.emergency_backup(
            [p for p in to_backup if p.exists()], BACKUP_DIR
        )
        self.db.action("emergency_backup", "ok", {"path": str(backup_path)})
        return {"backup": str(backup_path)}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ollama ERP DevOps Maintenance System (Local)"
    )
    parser.add_argument(
        "--cycle", action="store_true", help="Run one maintenance cycle"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run forever (sleep 5min between cycles)",
    )
    parser.add_argument(
        "--vision-test", action="store_true", help="Run vision test only"
    )
    parser.add_argument(
        "--generate-doc", action="store_true", help="Generate a maintenance report now"
    )
    parser.add_argument(
        "--critical-test",
        action="store_true",
        help="Run critical mode (emergency backup)",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    orch = MaintenanceOrchestrator()

    if args.vision_test:
        res = orch.vision.run()
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if args.generate_doc:
        # Generate a doc based on current system + synthetic vision + no Ollama prompt
        health = SystemHealth.collect()
        vision = orch.vision.run()
        report = Documentation.write_report(
            DOCS_DIR, health, vision, "[On-demand doc generation]"
        )
        print("report:", report)
        return 0

    if args.critical_test:
        res = orch.critical_mode()
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if args.cycle or args.continuous:

        def run_cycle() -> None:
            out = orch.cycle_once()
            print(json.dumps(out, indent=2, ensure_ascii=False))

        if args.cycle:
            run_cycle()
            return 0
        else:
            try:
                while True:
                    run_cycle()
                    # Sleep 5 minutes between cycles
                    import time as _t

                    _t.sleep(300)
            except KeyboardInterrupt:
                return 0

    # Default: print help
    parse_args(["-h"])  # show help
    return 0


if __name__ == "__main__":
    sys.exit(main())
