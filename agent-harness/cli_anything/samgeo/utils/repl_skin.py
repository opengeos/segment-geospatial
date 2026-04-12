"""Unified REPL skin for cli-anything-samgeo.

Provides banner, prompt, help, and styled message helpers for the interactive REPL.
"""

import os
import sys


class ReplSkin:
    """Themed REPL interface for cli-anything CLIs."""

    def __init__(self, name, version="0.1.0"):
        """Initialize the REPL skin.

        Args:
            name: CLI name (e.g., 'samgeo').
            version: CLI version string.
        """
        self.name = name
        self.version = version
        self._skill_path = self._find_skill_md()

    def _find_skill_md(self):
        """Auto-detect the SKILL.md path inside the package."""
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        skill_path = os.path.join(pkg_dir, "skills", "SKILL.md")
        if os.path.exists(skill_path):
            return skill_path
        return None

    def print_banner(self):
        """Print the startup banner."""
        width = 60
        line = "=" * width
        title = f" cli-anything-{self.name} v{self.version} "
        padded = title.center(width, "=")

        print(f"\n{padded}")
        print(f"  Geospatial Image Segmentation CLI")
        print(f"  Type 'help' for commands, 'quit' to exit")
        if self._skill_path:
            print(f"  Skill file: {self._skill_path}")
        print(f"{line}\n")

    def create_prompt_session(self):
        """Create a prompt_toolkit session with history and styling.

        Returns:
            PromptSession or None if prompt_toolkit is not available.
        """
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory

            history_file = os.path.expanduser(f"~/.cli_anything_{self.name}_history")
            return PromptSession(history=FileHistory(history_file))
        except ImportError:
            return None

    def get_input(self, session=None, project_name=None, modified=False):
        """Get user input with styled prompt.

        Args:
            session: prompt_toolkit PromptSession.
            project_name: Current project name for the prompt.
            modified: Whether the project has unsaved changes.

        Returns:
            str: User input line.
        """
        mod = "*" if modified else ""
        proj = f":{project_name}{mod}" if project_name else ""
        prompt_str = f"{self.name}{proj}> "

        if session:
            try:
                return session.prompt(prompt_str)
            except (EOFError, KeyboardInterrupt):
                return "quit"
        else:
            try:
                return input(prompt_str)
            except (EOFError, KeyboardInterrupt):
                return "quit"

    def help(self, commands):
        """Print formatted help listing.

        Args:
            commands: Dict mapping command names to descriptions.
        """
        max_len = max(len(k) for k in commands) if commands else 0
        print("\nAvailable commands:\n")
        for cmd, desc in sorted(commands.items()):
            print(f"  {cmd:<{max_len + 2}} {desc}")
        print()

    def success(self, message):
        """Print a success message.

        Args:
            message: Success message text.
        """
        print(f"  \u2713 {message}")

    def error(self, message):
        """Print an error message.

        Args:
            message: Error message text.
        """
        print(f"  \u2717 {message}", file=sys.stderr)

    def warning(self, message):
        """Print a warning message.

        Args:
            message: Warning message text.
        """
        print(f"  \u26a0 {message}")

    def info(self, message):
        """Print an info message.

        Args:
            message: Info message text.
        """
        print(f"  \u25cf {message}")

    def status(self, key, value):
        """Print a key-value status line.

        Args:
            key: Status key.
            value: Status value.
        """
        print(f"  {key}: {value}")

    def table(self, headers, rows):
        """Print a formatted table.

        Args:
            headers: List of column header strings.
            rows: List of row lists.
        """
        if not rows:
            print("  (no data)")
            return

        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
        print(f"  {header_line}")
        print(f"  {sep_line}")
        for row in rows:
            row_line = "  ".join(
                str(cell).ljust(widths[i]) for i, cell in enumerate(row)
            )
            print(f"  {row_line}")

    def progress(self, current, total, message=""):
        """Print a progress bar.

        Args:
            current: Current step.
            total: Total steps.
            message: Progress message.
        """
        pct = int(current / total * 100) if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        print(f"\r  [{bar}] {pct}% {message}", end="", flush=True)
        if current >= total:
            print()

    def print_goodbye(self):
        """Print the exit message."""
        print(f"\n  Goodbye from cli-anything-{self.name}!\n")
