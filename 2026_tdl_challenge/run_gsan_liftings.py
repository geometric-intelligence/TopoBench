import sys
from pathlib import Path

# Aggiungi cartella al path
_ROOT = Path.cwd().resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import run_challenge_grid, resolve_project_root, save_challenge_artifacts
import yaml

PROJECT_ROOT = resolve_project_root(_ROOT)
MODEL_CONFIG = "simplicial/gsan"

LIFTINGS = ["clique", "khop", "vietoris_rips"]

def main():
    lifting_yaml_path = PROJECT_ROOT / "configs" / "transforms" / "liftings" / "graph2simplicial_default.yaml"
    
    for lifting in LIFTINGS:
        print(f"\n======================================")
        print(f"🚀 INIZIO VALUTAZIONE CON LIFTING: {lifting}")
        print(f"======================================\n")
        
        config_content = f"""defaults:
  - /transforms/liftings/graph2simplicial@graph2simplicial_lifting: {lifting}
"""
        with open(lifting_yaml_path, "w") as f:
            f.write(config_content)
            
        results, study_id = run_challenge_grid(
            project_root=PROJECT_ROOT,
            model_config=MODEL_CONFIG,
            study_id=f"gsan_lifting_{lifting}",
            quiet=False,
        )
        
        output_paths = save_challenge_artifacts(
            results,
            model_config=MODEL_CONFIG,
            study_id=f"gsan_{lifting}",
        )
        print(f"✅ Risultati per {lifting} salvati in {output_paths['dir']}")

if __name__ == "__main__":
    main()
