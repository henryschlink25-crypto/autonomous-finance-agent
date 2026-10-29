"""
github_sync.py - GitHub Synchronisations- & Versionskontroll-Modul
Ermöglicht dem Benutzer, das Trading-System mit seinem GitHub-Account zu verbinden
und autonome Evolutions-Meilensteine, Sub-Agenten-Mutationen und Performance-Updates automatisch zu pushen.
"""

import os
import subprocess
import sys
from typing import Dict, Any, List, Tuple


class GitHubSyncManager:
    def __init__(self, repo_dir: str = None):
        if repo_dir is None:
            self.repo_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.repo_dir = repo_dir

    def _run_git(self, args: List[str]) -> Tuple[int, str, str]:
        try:
            res = subprocess.run(
                ["git"] + args,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            return res.returncode, res.stdout.strip(), res.stderr.strip()
        except Exception as e:
            return 1, "", str(e)

    def get_status(self) -> Dict[str, Any]:
        """Gibt den aktuellen Git- & Remote-Status zurück."""
        code, branch, _ = self._run_git(["branch", "--show-current"])
        _, remotes, _ = self._run_git(["remote", "-v"])
        _, status_short, _ = self._run_git(["status", "-s"])
        _, last_commit, _ = self._run_git(["log", "-1", "--oneline"])

        has_remote = "origin" in remotes
        remote_url = ""
        if has_remote:
            for line in remotes.splitlines():
                if "origin" in line and "(push)" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        remote_url = parts[1]

        return {
            "branch": branch or "main",
            "has_remote": has_remote,
            "remote_url": remote_url,
            "uncommitted_files": len(status_short.splitlines()) if status_short else 0,
            "last_commit": last_commit or "Kein Commit vorhanden",
        }

    def set_remote(self, github_repo_url: str) -> bool:
        """Verknüpft das lokale Repository mit der angegebenen GitHub URL."""
        if not github_repo_url:
            return False
        # Prüfen ob origin schon existiert
        _, remotes, _ = self._run_git(["remote"])
        if "origin" in remotes.split():
            code, _, _ = self._run_git(["remote", "set-url", "origin", github_repo_url])
        else:
            code, _, _ = self._run_git(["remote", "add", "origin", github_repo_url])
        return code == 0

    def commit_evolution_checkpoint(self, message: str = "") -> bool:
        """Erstellt einen sauberen Commit für neue Evolutionsschritte / Reflexions-Ergebnisse."""
        if not message:
            message = "adas: checkpoint - updated evolutionary state & swarm reflections"
        # .env wird dank .gitignore niemals committed
        self._run_git(["add", "."])
        code, out, _ = self._run_git(["commit", "-m", message])
        return code == 0

    def push_to_github(self) -> Tuple[bool, str]:
        """Pusht den aktuellen Stand zu GitHub."""
        status = self.get_status()
        if not status["has_remote"]:
            return False, "Kein Remote-Repository ('origin') konfiguriert. Bitte erstelle ein GitHub Repo und hinterlege die URL."
        code, out, err = self._run_git(["push", "-u", "origin", status["branch"]])
        if code == 0:
            return True, f"Erfolgreich zu {status['remote_url']} gepusht!"
        return False, err or out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sync = GitHubSyncManager()
    st = sync.get_status()
    print("=== [GitHub Sync Manager Status] ===")
    print(f"Aktueller Branch: {st['branch']}")
    print(f"Remote verknuepft: {'Ja (' + st['remote_url'] + ')' if st['has_remote'] else 'Nein'}")
    print(f"Letzter Commit:   {st['last_commit']}")
    print(f"Offene Dateien:   {st['uncommitted_files']}")
    if len(sys.argv) > 1 and sys.argv[1] == "--set-remote" and len(sys.argv) > 2:
        url = sys.argv[2]
        if sync.set_remote(url):
            print(f"[OK] Remote erfolgreich gesetzt auf: {url}")
        else:
            print(f"[ERR] Fehler beim Setzen des Remotes auf: {url}")

