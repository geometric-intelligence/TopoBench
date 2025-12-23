# master_sweep.sh
# This script orchestrates the sequential execution of individual lifting sweep scripts.

# --- Configuration ---
# List of individual sweep scripts to run.
# Ensure these scripts are executable (chmod +x)
SWEEP_SCRIPTS=(
    "scripts/sweep_knn.sh"
    "scripts/sweep_khop.sh"
    "scripts/sweep_ekhop.sh"
    "scripts/sweep_fr.sh"
    # "scripts/sweep_kernel.sh"  --- DISABLED FOR NOW ---
    "scripts/sweep_mapper.sh"
    "scripts/sweep_modularity.sh"
)

# Loop through the list of scripts and execute each one sequentially
for script_path in "${SWEEP_SCRIPTS[@]}"; do
    
    # Check if the script exists and is executable
    if [ ! -x "$script_path" ]; then
        echo "ERROR: Script not found or not executable: $script_path. Skipping."
        continue
    fi
    
    echo "STARTING: $script_path"
    
    "$script_path"
    
    if [ $? -ne 0 ]; then
        echo "WARNING: $script_path failed (Exit Code: $?). Continuing to next script."
    else
        echo "FINISHED: $script_path"
    fi

done

# --- Final Cleanup ---
echo "=========================================================="
echo " All sweep scripts have finished execution."
echo "=========================================================="