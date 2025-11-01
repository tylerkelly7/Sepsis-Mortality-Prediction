import os
import glob, joblib
from pathlib import Path

def get_repo_root() -> str:
    """
    Return the absolute path to the repository root
    (the directory containing setup.py).
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(relative_path: str) -> Path:
    """
    Resolve a relative path (from project root) into a full absolute Path object.
    Always returns a pathlib.Path (not a string).

    Example:
        resolve_path("data/processed") ->
        "C:/Users/.../Masters-Thesis/data/processed"
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "Masters-Thesis":  # dynamically detect project root
            project_root = parent
            break
    else:
        raise FileNotFoundError("❌ Could not locate project root 'Masters-Thesis'.")
    
    abs_path = project_root / relative_path
    return abs_path

def load_latest_artifact(patterns):
    """
    Return the most recent file matching ANY of the provided glob patterns.
    Patterns should include a dated subdir wildcard like '*/'.
    """
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(str(resolve_path(p))))
    if not candidates:
        raise FileNotFoundError(f"No model artifacts found for patterns:\n  " + "\n  ".join(patterns))
    # sort by mtime (fallback to lexicographic if needed)
    candidates = sorted(candidates, key=lambda p: Path(p).stat().st_mtime)
    latest = candidates[-1]
    print("📦 Using artifact:", latest)
    return latest

def save_fig(fig, name, tight=True, dpi=300):
    """
    Save matplotlib figure to reports/figures/exploration using resolve_path().
    """
    # define export directory via resolve_path
    fig_dir = resolve_path("reports/figures/exploration")
    os.makedirs(fig_dir, exist_ok=True)
    
    path = fig_dir / f"{name}.png"

    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"✅ Saved figure: {path}")