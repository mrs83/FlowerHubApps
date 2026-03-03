#!/bin/bash

# Utility script to login to Flower Supergrid and publish apps interactively.

echo "Initiating Flower login..."
flwr login supergrid

APPS_DIR="apps"

# Check if apps directory exists
if [ ! -d "$APPS_DIR" ]; then
    echo "Error: Directory '$APPS_DIR' not found."
    exit 1
fi

echo "--------------------------------------------------"
echo "Starting interactive app publication process"
echo "--------------------------------------------------"

# Iterate through all subdirectories in 'apps'
for app in "$APPS_DIR"/*/; do
    # Remove trailing slash for cleaner output
    app_path="${app%/}"
    app_name=$(basename "$app_path")

    # Skip if it's not a directory or if it's the .venv directory
    if [ ! -d "$app_path" ] || [ "$app_name" == ".venv" ]; then
        continue
    fi

    # Confirm with the user
    read -p "Publish app '$app_name'? (y/n): " choice
    case "$choice" in 
        y|Y ) 
            echo "Publishing $app_name..."
            # Execute the publish command. 
            # Note: We pass the directory path relative to the root or however flwr expects it.
            # Usually flwr app publish [dirname]
            flwr app publish "$app_path"
            ;;
        * ) 
            echo "Skipping $app_name."
            ;;
    esac
    echo "--------------------------------------------------"
done

echo "Finished processing all apps."
