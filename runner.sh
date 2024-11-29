#!/bin/bash

# Fixed base command
BASE_CMD="python3 main.py --root_path data/"

# Options to iterate over
#DATASETS=("circuits" "eurosat")
DATASETS=("circuits")
SEEDS=(1 3 5)

# Enable combinations (all combinations of --enable_lora and --enable_BitFit)
# ENABLE_COMBINATIONS=(
#     "--enable_lora"             # Only lora enabled
#     "--enable_BitFit"           # Only BitFit enabled
#     "--enable_lora --enable_BitFit" # Both lora and BitFit enabled
# )

ENABLE_COMBINATIONS=(
    "--enable_MetaAdapteR"
    "--enable_lora --enable_BitFit --enable_MetaAdapter"
)

# Loop through datasets, seeds, and enable combinations
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Construct save path
        SAVE_PATH="ckpt/seed_${SEED}"
        if [ "$DATASET" == "circuits" ]; then
            SHOTS_VALUES=(4 8)
        else
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
                
                # Construct and execute the command
                CMD="${BASE_CMD} --dataset $DATASET --seed $SEED --shots $SHOTS $ENABLE_FLAGS --save_path $SAVE_PATH --filename $FILENAME"
                $CMD
            done
        done
    done
done