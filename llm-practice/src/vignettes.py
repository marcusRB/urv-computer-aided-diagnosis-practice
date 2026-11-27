"""Parse Clinical_Vignettes_26_27.txt and draw reproducible samples."""
from pathlib import Path
import random


def load_vignettes(path):
    """Return clean 4-line Patient/Doctor/Patient/Doctor dialogues.

    id = index of the block in the file (stable, so it can be cited in the report).
    gold = the final Doctor line (reference diagnosis statement).
    """
    raw = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\r", "")
    out = []
    for idx, block in enumerate(b for b in raw.split("\n\n") if b.strip()):
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if (len(lines) == 4 and lines[0].startswith("Patient:") and lines[1].startswith("Doctor:")
                and lines[2].startswith("Patient:") and lines[3].startswith("Doctor:")):
            f = lambda s: s.split(":", 1)[1].strip()
            out.append({"id": idx, "p1": f(lines[0]), "d1": f(lines[1]),
                        "p2": f(lines[2]), "gold": f(lines[3])})
    return out


def sample_vignettes(vignettes, n_core=5, n_extra=0, seed=2026):
    """Seeded sample. First n_core = 'core' set (the 5 required by the practice),
    next n_extra = 'extra' set (for statistics). Same draw for every model."""
    rng = random.Random(seed)
    picked = rng.sample(vignettes, n_core + n_extra)
    for i, v in enumerate(picked):
        v["set"] = "core" if i < n_core else "extra"
    return picked
