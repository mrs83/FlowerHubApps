#!/bin/bash

# Utility script to list all apps in the 'apps' directory and execute a sequence of commands for each.
# Commands: python3 -m venv .venv && source .venv/bin/activate && pip install -e . && flwr run && deactivate && rm -rf .venv

APPS_DIR="apps"

# Arrays to track status
INSTALLED_APPS=()
WORKED_APPS=()
FAILED_APPS=()
FAILURE_OCCURRED=false

# Apps to exclude from execution
EXCLUDE_APPS=("phi-4-nlp")

# Check if apps directory exists
if [ ! -d "$APPS_DIR" ]; then
    echo "Error: Directory '$APPS_DIR' not found."
    exit 1
fi

# Iterate through all subdirectories in 'apps'
for app in "$APPS_DIR"/*/; do
    # Remove trailing slash for cleaner output
    app_path="${app%/}"
    app_name=$(basename "$app_path")

    # Skip if it's in the exclusion list
    if [[ " ${EXCLUDE_APPS[@]} " =~ " ${app_name} " ]]; then
        echo "Skipping excluded app: $app_name"
        continue
    fi

    # Skip if it's not a directory or if it's the .venv directory itself
    if [ ! -d "$app_path" ] || [ "$app_name" == ".venv" ]; then
        continue
    fi

    echo "--------------------------------------------------"
    echo "Processing app: $app_name"
    echo "--------------------------------------------------"

    # Navigate into the app directory
    cd "$app_path" || { echo "Failed to enter $app_path"; FAILED_APPS+=("$app_name (CD failed)"); continue; }

    # Setup venv and execute sequence with cleanup
    echo "Setting up virtual environment in $app_name..."
    python3 -m venv .venv
    
    # Track status within the execution block
    INSTALL_SUCCESS=false
    RUN_SUCCESS=false

    # Use a subshell or a block to ensure we can cleanup regardless of what happens inside
    # We use temporary files to pass status out of the subshell if needed, 
    # but since we are in a loop, we can just check exit codes.
    
    if ( source .venv/bin/activate && pip install -e . ); then
        INSTALL_SUCCESS=true
        INSTALLED_APPS+=("$app_name")
        echo "Installation successful for $app_name."
        
        if ( source .venv/bin/activate && flwr run ); then
            RUN_SUCCESS=true
            WORKED_APPS+=("$app_name")
            echo "'flwr run' successful for $app_name."
        else
            FAILED_APPS+=("$app_name (flwr run failed)")
            echo "'flwr run' failed for $app_name."
            FAILURE_OCCURRED=true
        fi
    else
        FAILED_APPS+=("$app_name (Installation failed)")
        echo "Installation failed for $app_name."
        FAILURE_OCCURRED=true
    fi
    
    # The following commands run regardless of the success/failure of the block above
    echo "Cleaning up $app_name..."
    # Check if we are in a venv before deactivating
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate
    fi
    rm -rf .venv

    # Return to the original directory
    cd - > /dev/null || exit

    # Stop at the first failure
    if [ "$FAILURE_OCCURRED" = true ]; then
        echo "Failure detected. Stopping execution."
        break
    fi
done

echo ""
echo "=================================================="
echo "                FINAL REPORT                      "
echo "=================================================="
echo "Apps Installed Successfully:"
for app in "${INSTALLED_APPS[@]}"; do
    echo "  - $app"
done

echo ""
echo "Apps where 'flwr run' Worked:"
for app in "${WORKED_APPS[@]}"; do
    echo "  - $app"
done

echo ""
echo "Apps that Failed (at any stage):"
for app in "${FAILED_APPS[@]}"; do
    echo "  - $app"
done
echo "=================================================="
echo "Finished processing all apps."
