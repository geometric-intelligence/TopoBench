"""
🚀 TopoBench Experiment Launcher

A comprehensive Streamlit app for:
- Auto-discovering all models from configs/model
- Grid search with Hydra --multirun
- Equal distribution of multirun blocks across GPUs
- Live experiment monitoring

Run with: streamlit run experiment_launcher.py
"""

import streamlit as st
import subprocess
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# ============================================================================
# Auto-Discovery of Models
# ============================================================================

def discover_models(config_dir: str = "configs/model") -> Dict[str, Dict]:
    """Auto-discover all model configurations from the config directory."""
    models = {}
    config_path = Path(config_dir)
    
    if not config_path.exists():
        return models
    
    for domain_dir in config_path.iterdir():
        if domain_dir.is_dir():
            domain = domain_dir.name
            for yaml_file in domain_dir.glob("*.yaml"):
                model_name = yaml_file.stem
                model_key = f"{domain}/{model_name}"
                
                try:
                    with open(yaml_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    models[model_key] = {
                        "path": str(yaml_file),
                        "domain": domain,
                        "name": model_name,
                        "display_name": f"{model_name.upper()} ({domain})",
                        "config": config,
                        "has_neighborhoods": "neighborhoods" in str(config),
                        "hyperparameters": extract_hyperparameters(config),
                    }
                except Exception:
                    pass
    
    return models

def extract_hyperparameters(config: Dict) -> Dict[str, Dict]:
    """Extract tunable hyperparameters from a model config."""
    hparams = {}
    
    if "feature_encoder" in config:
        fe = config["feature_encoder"]
        if "out_channels" in fe:
            hparams["model.feature_encoder.out_channels"] = {
                "display": "Hidden Dimension",
                "options": [16, 32, 64, 128, 256],
                "default": [64],
            }
        if "proj_dropout" in fe:
            hparams["model.feature_encoder.proj_dropout"] = {
                "display": "Encoder Dropout",
                "options": [0.0, 0.1, 0.2, 0.25, 0.5],
                "default": [0.0],
            }
    
    if "backbone" in config:
        bb = config["backbone"]
        if "layers" in bb:
            hparams["model.backbone.layers"] = {
                "display": "Backbone Layers",
                "options": [1, 2, 3],
                "default": [1],
            }
        if "activation" in bb:
            hparams["model.backbone.activation"] = {
                "display": "Activation",
                "options": ["relu", "tanh", "elu"],
                "default": ["relu"],
            }
    
    return hparams

# ============================================================================
# Neighborhood Configuration
# ============================================================================

NEIGHBORHOODS = {
    "1-up_adjacency-0": {"desc": "Posts↔Posts (Users)", "edges": 257_005_682, "ok": False, "type": "adjacency"},
    "2-up_adjacency-0": {"desc": "Posts↔Posts (Interactions)", "edges": 1_208, "ok": True, "type": "adjacency"},
    "3-up_adjacency-0": {"desc": "Posts↔Posts (Communities)", "edges": 581_230_654, "ok": False, "type": "adjacency"},
    "1-down_incidence-1": {"desc": "Users→Posts", "edges": 39_079, "ok": True, "type": "incidence"},
    "2-down_incidence-2": {"desc": "Interactions→Posts", "edges": 50_000, "ok": True, "type": "incidence"},
    "3-down_incidence-3": {"desc": "Communities→Posts", "edges": 39_079, "ok": True, "type": "incidence"},
    "4-down_incidence-4": {"desc": "Semantic→Posts", "edges": 39_079, "ok": True, "type": "incidence"},
}

MAGA_STRUCTURE = {
    "Rank 0 - Posts": {"count": 39_079, "features": 1024, "color": "#FF6B6B", "desc": "Twitter posts"},
    "Rank 1 - Users": {"count": 39, "features": 1024, "color": "#4ECDC4", "desc": "User bios"},
    "Rank 2 - Interactions": {"count": 252, "features": 1024, "color": "#45B7D1", "desc": "Threads"},
    "Rank 3 - Communities": {"count": 7, "features": 1024, "color": "#96CEB4", "desc": "User clusters"},
    "Rank 4 - Semantic": {"count": 74, "features": 74, "color": "#FFEAA7", "desc": "Topic clusters"},
}

# ============================================================================
# Command Generation
# ============================================================================

def generate_multirun_command(
    model_path: str,
    params: Dict[str, List],
    neighborhoods: List[str],
    gpu: int,
    wandb_project: str,
    tag: str
) -> str:
    """Generate a single Hydra --multirun command."""
    cmd_parts = ["python -m topobench"]
    cmd_parts.append("dataset=hypergraph/maga_arlequin")
    cmd_parts.append(f"model={model_path}")
    
    # Neighborhoods
    if neighborhoods:
        nbhd_str = "[" + ",".join(neighborhoods) + "]"
        cmd_parts.append(f"model.backbone.neighborhoods={nbhd_str}")
    
    # Parameters - join multiple values with comma for Hydra multirun
    for key, values in params.items():
        if isinstance(values, list) and len(values) > 0:
            if len(values) > 1:
                val_str = ",".join(str(v) for v in values)
            else:
                val_str = str(values[0])
            cmd_parts.append(f"{key}={val_str}")
        elif not isinstance(values, list):
            cmd_parts.append(f"{key}={values}")
    
    # Fixed params
    cmd_parts.append(f"trainer.devices=[{gpu}]")
    cmd_parts.append(f"logger.wandb.project={wandb_project}")
    cmd_parts.append(f"tags=[{tag}]")
    cmd_parts.append("--multirun")
    
    return " \\\n    ".join(cmd_parts)

def count_experiments_in_multirun(params: Dict[str, List]) -> int:
    """Count how many experiments a multirun command will generate."""
    total = 1
    for values in params.values():
        if isinstance(values, list):
            total *= max(len(values), 1)
    return total

# ============================================================================
# Streamlit App
# ============================================================================

def main():
    st.set_page_config(
        page_title="TopoBench Launcher",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .rank-box { padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 5px solid; }
    .gpu-box { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
               padding: 15px; border-radius: 10px; margin: 5px; text-align: center; }
    .block-box { background-color: #1a1a2e; padding: 15px; border-radius: 10px; 
                 margin: 10px 0; border-left: 4px solid; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚀 TopoBench Experiment Launcher")
    
    models = discover_models()
    if not models:
        st.error("No models found!")
        return
    
    # ========== Sidebar ==========
    with st.sidebar:
        st.header("📊 MAGA Dataset")
        for rank, info in MAGA_STRUCTURE.items():
            st.markdown(f"""
            <div class="rank-box" style="background-color: {info['color']}22; border-color: {info['color']};">
                <strong style="color: {info['color']};">{rank}</strong><br>
                <span style="font-size: 1.2em; font-weight: bold;">{info['count']:,}</span> cells<br>
                <small>{info['desc']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown(f"**📦 {len(models)} Models Available**")
        
        domains = {}
        for key, info in models.items():
            d = info["domain"]
            domains[d] = domains.get(d, []) + [info["name"]]
        
        for domain, model_list in sorted(domains.items()):
            color = {"hypergraph": "#4ECDC4", "graph": "#FF6B6B", "cell": "#45B7D1",
                     "simplicial": "#96CEB4", "combinatorial": "#FFEAA7"}.get(domain, "#888")
            st.markdown(f"<small style='color:{color}'><b>{domain}</b>: {', '.join(model_list)}</small>", 
                       unsafe_allow_html=True)
    
    # ========== Main Content ==========
    tab1, tab2, tab3 = st.tabs(["⚙️ Grid Search", "📡 Monitor", "📜 History"])
    
    with tab1:
        st.header("⚙️ Grid Search Configuration")
        st.info("💡 **Strategy:** Each GPU runs ONE `--multirun` block. Select a parameter to split by, and each unique value becomes a separate block assigned to a GPU.")
        
        # Model selection
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_model = st.selectbox(
                "🔧 Model",
                sorted(models.keys()),
                format_func=lambda x: f"{models[x]['name'].upper()} ({models[x]['domain']})"
            )
        with col2:
            model_info = models[selected_model]
            st.caption(f"Config: `{selected_model}` | Neighborhoods: {'✅' if model_info['has_neighborhoods'] else '❌'}")
        
        st.divider()
        
        # ===== Neighborhoods =====
        if model_info["has_neighborhoods"]:
            st.subheader("🔗 Neighborhoods")
            is_hyp = "hyp" in selected_model.lower() or "allset" in selected_model.lower()
            nbhd_type = "incidence" if is_hyp else "adjacency"
            st.caption(f"Model type: **{nbhd_type}**")
            
            filtered = {k: v for k, v in NEIGHBORHOODS.items() if v["type"] == nbhd_type}
            selected_nbhds = []
            
            cols = st.columns(len(filtered))
            for i, (key, info) in enumerate(filtered.items()):
                with cols[i]:
                    color = "#00ff88" if info["ok"] else "#ff4757"
                    icon = "✅" if info["ok"] else "❌"
                    st.markdown(f"<div style='background:{color}20;padding:8px;border-radius:5px;text-align:center;border:1px solid {color}'><b>{key.split('-')[0]}</b></div>", unsafe_allow_html=True)
                    if st.checkbox(f"{icon}", disabled=not info["ok"], key=f"nb_{key}"):
                        selected_nbhds.append(key)
                    st.caption(info["desc"])
        else:
            selected_nbhds = []
        
        st.divider()
        
        # ===== Split Parameter Selection =====
        st.subheader("🔀 Block Distribution")
        st.markdown("**Choose how to split experiments into GPU blocks:**")
        
        split_options = ["neighborhoods", "seeds"]
        split_by = st.radio(
            "Split multirun blocks by:",
            split_options,
            horizontal=True,
            help="Each unique value of this parameter gets its own GPU"
        )
        
        st.divider()
        
        # ===== Hyperparameters =====
        st.subheader("🎛️ Hyperparameters (Grid Search)")
        
        all_params = {}
        
        # Model-specific params
        if model_info["hyperparameters"]:
            cols = st.columns(3)
            for i, (key, cfg) in enumerate(model_info["hyperparameters"].items()):
                with cols[i % 3]:
                    all_params[key] = st.multiselect(cfg["display"], cfg["options"], cfg["default"], key=f"p_{key}")
        
        # Training params
        st.markdown("**Training:**")
        cols = st.columns(4)
        with cols[0]:
            all_params["optimizer.parameters.lr"] = st.multiselect("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], [0.001])
        with cols[1]:
            all_params["optimizer.parameters.weight_decay"] = st.multiselect("Weight Decay", [0.0, 0.0001, 0.0005], [0.0])
        with cols[2]:
            all_params["dataset.split_params.data_seed"] = st.multiselect("Seeds", [1, 2, 3, 4, 5, 42], [1, 3, 5])
        with cols[3]:
            all_params["model.readout.pooling_type"] = st.multiselect("Pooling", ["mean", "sum", "max"], ["mean"])
        
        cols = st.columns(3)
        with cols[0]:
            all_params["trainer.max_epochs"] = st.number_input("Max Epochs", 100, 2000, 1000)
        with cols[1]:
            all_params["trainer.min_epochs"] = st.number_input("Min Epochs", 50, 500, 250)
        with cols[2]:
            all_params["callbacks.early_stopping.patience"] = st.number_input("Patience", 10, 300, 100)
        
        st.divider()
        
        # ===== GPU Selection =====
        st.subheader("🖥️ GPU Assignment")
        
        cols = st.columns(3)
        with cols[0]:
            available_gpus = st.multiselect("Available GPUs", [0, 1, 2, 3], [0, 1, 2, 3])
        with cols[1]:
            wandb_project = st.text_input("WandB Project", "MAGA_GridSearch")
        with cols[2]:
            tag = st.text_input("Tag", "grid")
        
        st.divider()
        
        # ===== Generate Blocks =====
        st.subheader("📦 Generated Multirun Blocks")
        
        # Determine blocks based on split parameter
        if split_by == "neighborhoods" and selected_nbhds:
            blocks = [{"neighborhoods": [n], "name": n} for n in selected_nbhds]
        elif split_by == "seeds" and all_params.get("dataset.split_params.data_seed"):
            seeds = all_params["dataset.split_params.data_seed"]
            blocks = [{"seed": s, "name": f"seed={s}"} for s in seeds]
        else:
            blocks = [{"name": "all"}]
        
        num_blocks = len(blocks)
        num_gpus = len(available_gpus)
        
        # Check feasibility
        if num_blocks > num_gpus:
            st.warning(f"⚠️ {num_blocks} blocks but only {num_gpus} GPUs. Only first {num_gpus} blocks will run.")
            blocks = blocks[:num_gpus]
        elif num_blocks < num_gpus:
            st.info(f"ℹ️ {num_blocks} blocks for {num_gpus} GPUs. {num_gpus - num_blocks} GPU(s) will be idle.")
        
        # Generate commands for each block
        commands = []
        for i, block in enumerate(blocks):
            gpu = available_gpus[i]
            
            # Build params for this block
            block_params = {k: v for k, v in all_params.items()}
            
            # Handle neighborhoods
            if "neighborhoods" in block:
                nbhds = block["neighborhoods"]
            else:
                nbhds = selected_nbhds if selected_nbhds else None
            
            # Handle seeds
            if "seed" in block:
                block_params["dataset.split_params.data_seed"] = [block["seed"]]
            
            # Count experiments in this block
            exp_count = count_experiments_in_multirun(block_params)
            if nbhds and split_by != "neighborhoods":
                exp_count *= len(nbhds) if isinstance(nbhds, list) else 1
            
            cmd = generate_multirun_command(
                selected_model,
                block_params,
                nbhds,
                gpu,
                wandb_project,
                tag
            )
            
            commands.append({
                "block": block,
                "gpu": gpu,
                "command": cmd,
                "experiments": exp_count
            })
        
        # Display blocks
        total_experiments = sum(c["experiments"] for c in commands)
        
        cols = st.columns(4)
        cols[0].metric("📦 Blocks", len(commands))
        cols[1].metric("🖥️ GPUs Used", len(commands))
        cols[2].metric("🔢 Total Experiments", total_experiments)
        cols[3].metric("⏱️ Est. Time", f"~{total_experiments * 5 / 60:.1f}h")
        
        # Show each block
        for cmd_info in commands:
            gpu = cmd_info["gpu"]
            color = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"][gpu % 4]
            
            st.markdown(f"""
            <div class="block-box" style="border-color: {color};">
                <span style="color: {color}; font-size: 1.2em; font-weight: bold;">
                    🖥️ GPU {gpu} — {cmd_info['block']['name']}
                </span>
                <span style="float: right; color: #888;">
                    {cmd_info['experiments']} experiments
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"View command for GPU {gpu}"):
                st.code(cmd_info["command"], language="bash")
        
        st.divider()
        
        # ===== Generate Script & Launch =====
        script = "#!/bin/bash\n\n"
        script += f"# TopoBench Grid Search - {selected_model}\n"
        script += f"# {len(commands)} blocks, {total_experiments} total experiments\n"
        script += f"# Split by: {split_by}\n"
        script += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        for cmd_info in commands:
            script += f"# Block: {cmd_info['block']['name']} -> GPU {cmd_info['gpu']} ({cmd_info['experiments']} experiments)\n"
            script += cmd_info["command"].replace(" \\\n    ", " ") + " &\n\n"
        
        script += "wait\necho '✅ All blocks completed!'\n"
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download Script", script, f"grid_{selected_model.replace('/', '_')}.sh")
        
        with col2:
            if st.button("🚀 Launch All Blocks", type="primary"):
                if not commands:
                    st.error("No commands to launch!")
                else:
                    for cmd_info in commands:
                        flat_cmd = cmd_info["command"].replace(" \\\n    ", " ")
                        proc = subprocess.Popen(
                            flat_cmd,
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            cwd=Path(__file__).parent
                        )
                        st.success(f"✅ GPU {cmd_info['gpu']}: Launched (PID {proc.pid})")
                    st.balloons()
    
    # ========== Monitor Tab ==========
    with tab2:
        st.header("📡 Experiment Monitor")
        
        if st.button("🔄 Refresh"):
            st.rerun()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Running Processes")
            try:
                result = subprocess.run(
                    "ps aux | grep 'python -m topobench' | grep -v grep",
                    shell=True, capture_output=True, text=True
                )
                if result.stdout:
                    processes = result.stdout.strip().split('\n')
                    st.success(f"🟢 {len(processes)} process(es) running")
                    for proc in processes:
                        parts = proc.split()
                        if len(parts) > 1:
                            st.code(f"PID: {parts[1]}")
                else:
                    st.info("No experiments running")
            except Exception as e:
                st.error(str(e))
        
        with col2:
            st.subheader("GPU Status")
            try:
                result = subprocess.run(
                    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
                    shell=True, capture_output=True, text=True
                )
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_id, mem_used, mem_total, util = parts
                            color = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"][int(gpu_id) % 4]
                            st.markdown(f"""
                            <div class="gpu-box" style="border: 2px solid {color};">
                                <div style="color: {color}; font-weight: bold;">GPU {gpu_id}</div>
                                <div>{util}% | {mem_used}/{mem_total}MB</div>
                            </div>
                            """, unsafe_allow_html=True)
            except:
                st.warning("Could not query GPUs")
    
    # ========== History Tab ==========
    with tab3:
        st.header("📜 Experiment History")
        
        log_dir = Path("logs/train/multiruns")
        if log_dir.exists():
            runs = sorted(log_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:15]
            
            for run_dir in runs:
                multirun_yaml = run_dir / "multirun.yaml"
                if multirun_yaml.exists():
                    exp_count = len([d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()])
                    with st.expander(f"📁 {run_dir.name} ({exp_count} experiments)"):
                        try:
                            with open(multirun_yaml) as f:
                                config = yaml.safe_load(f)
                            if "hydra" in config:
                                overrides = config["hydra"].get("overrides", {}).get("task", [])
                                st.code("\n".join(overrides[:15]), language="yaml")
                        except:
                            pass
        else:
            st.info("No history found")

if __name__ == "__main__":
    main()
