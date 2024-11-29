#!/bin/bash

# Fixed base command
BASE_CMD="python3 main.py --root_path data/"

# Options to iterate over
DATASETS=("circuits" "eurosat")
SEEDS=(1 3 5)

# Enable combinations (all combinations of --enable_lora and --enable_BitFit)
ENABLE_COMBINATIONS=(
    "--enable_lora"             # Only lora enabled
    "--enable_BitFit"           # Only BitFit enabled
    "--enable_lora --enable_BitFit" # Both lora and BitFit enabled
)

# Loop through datasets, seeds, enable combinations
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # If dataset is 'circuits', test with shots 4, 8, and 16
        if [ "$DATASET" == "circuits" ]; then
            SHOTS_VALUES=(4 8 16)
        else
            # For other datasets, use the default shot value 16
            SHOTS_VALUES=(16)
        fi
        
        for SHOTS in "${SHOTS_VALUES[@]}"; do
            for ENABLE_FLAGS in "${ENABLE_COMBINATIONS[@]}"; do
                
                # Construct filename based on enabled options
                FILENAME="clip"
                if [[ "$ENABLE_FLAGS" == *"--enable_lora"* ]]; then
                    FILENAME+="_lora"
                fi
                if [[ "$ENABLE_FLAGS" == *"--enable_BitFit"* ]]; then
                    FILENAME+="_bitfit"
                fi
                FILENAME+="_${DATASET}_${SHOTS}shots"
                
                # Construct save path
                CKPT_PATH="ckpt/seed_${SEED}/$FILENAME.pt"
                
                # Construct the full command
                CMD="${BASE_CMD} --dataset $DATASET --seed $SEED --shots $SHOTS $ENABLE_FLAGS --eval_only --load_ckpt $CKPT_PATH"

                OUTPUT=$($CMD 2>&1)
                
                # Run the command and capture the final result, then append to results file
                RESULT=$(echo "$OUTPUT" | tail -n 1) # Get the last line of output from Python program
                
                # Append the result to the file with the command and result
                echo "dataset $DATASET, seed $SEED, $ENABLE_FLAGS, shots $SHOTS, checkpoint $CKPT_PATH, $RESULT" >> evaluation_results.txt
            done
        done
    done
done