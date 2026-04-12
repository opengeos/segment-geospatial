"""Session management: undo/redo, history tracking, state persistence."""

import copy
import json
import os
from datetime import datetime, timezone


class Session:
    """Manages stateful CLI sessions with undo/redo capability.

    The session wraps a project dict and tracks all mutations for undo/redo.
    """

    def __init__(self, project=None):
        """Initialize a session.

        Args:
            project: Initial project state dict. If None, starts empty.
        """
        self._project = project or {}
        self._undo_stack = []
        self._redo_stack = []
        self._session_file = None

    @property
    def project(self):
        """Get the current project state."""
        return self._project

    @project.setter
    def project(self, value):
        """Set the project state (pushes old state to undo stack)."""
        self._push_undo()
        self._project = value
        self._redo_stack.clear()

    def mutate(self, key, value):
        """Mutate a single project key with undo support.

        Args:
            key: The project dict key to set.
            value: The new value.
        """
        self._push_undo()
        self._project[key] = value
        self._redo_stack.clear()

    def update_project(self, updates):
        """Apply multiple updates to the project with undo support.

        Args:
            updates: Dict of key-value pairs to update.
        """
        self._push_undo()
        self._project.update(updates)
        self._redo_stack.clear()

    def undo(self):
        """Undo the last mutation.

        Returns:
            bool: True if undo was performed, False if nothing to undo.
        """
        if not self._undo_stack:
            return False
        self._redo_stack.append(copy.deepcopy(self._project))
        self._project = self._undo_stack.pop()
        return True

    def redo(self):
        """Redo the last undone mutation.

        Returns:
            bool: True if redo was performed, False if nothing to redo.
        """
        if not self._redo_stack:
            return False
        self._undo_stack.append(copy.deepcopy(self._project))
        self._project = self._redo_stack.pop()
        return True

    @property
    def can_undo(self):
        """Whether undo is available."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self):
        """Whether redo is available."""
        return len(self._redo_stack) > 0

    def get_status(self):
        """Get session status info.

        Returns:
            dict: Session status.
        """
        return {
            "has_project": bool(self._project),
            "project_name": self._project.get("name"),
            "undo_depth": len(self._undo_stack),
            "redo_depth": len(self._redo_stack),
            "history_count": len(self._project.get("history", [])),
            "session_file": self._session_file,
        }

    def get_history(self, limit=None):
        """Get the operation history.

        Args:
            limit: Maximum number of entries to return. None for all.

        Returns:
            list: History entries, most recent first.
        """
        history = list(reversed(self._project.get("history", [])))
        if limit:
            history = history[:limit]
        return history

    def save_session(self, path=None):
        """Save the session state to a JSON file.

        Args:
            path: File path. If None, uses the previously set path.

        Returns:
            str: The path saved to.
        """
        path = path or self._session_file
        if not path:
            raise ValueError("No session file path specified.")

        path = os.path.abspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        session_data = {
            "project": {
                k: v for k, v in self._project.items() if not k.startswith("_")
            },
            "undo_count": len(self._undo_stack),
            "redo_count": len(self._redo_stack),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(path, "w") as f:
            json.dump(session_data, f, indent=2)

        self._session_file = path
        return path

    def load_session(self, path):
        """Load session state from a JSON file.

        Args:
            path: Path to the session file.

        Returns:
            dict: The loaded project state.
        """
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        self._project = data.get("project", {})
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._session_file = path
        return self._project

    def _push_undo(self):
        """Push current state to the undo stack."""
        self._undo_stack.append(copy.deepcopy(self._project))
        # Limit undo depth to prevent memory issues
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
