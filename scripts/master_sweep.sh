# master_sweep.sh
# This script orchestrates the sequential execution of individual lifting sweep scripts.

# --- Configuration ---
# List of individual sweep scripts to run.
# Ensure these scripts are executable (chmod +x)
SWEEP_SCRIPTS=(
    "./sweep_knn.sh"
    "./sweep_khop.sh"
    "./sweep_ekhop.sh"
    "./sweep_fr.sh"
    # "./sweep_kernel.sh"  --- DISABLED FOR NOW ---
    "./sweep_mapper.sh"
    "./sweep_modularity.sh"
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