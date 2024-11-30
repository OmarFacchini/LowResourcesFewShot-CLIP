#!/bin/bash

# Fixed base command
BASE_CMD="python3 main.py --root_path data/"

# Options to iterate over
#DATASETS=("circuits" "eurosat")
DATASETS=("eurosat")
SEEDS=(1 3 5)


# total combinations possible (only use for circuits due to breaking loss)
# ENABLE_COMBINATIONS=(
#     "--enable_lora"                                                               # Only lora enabled
#     "--enable_BitFit"                                                             # Only BitFit enabled
#     "--enable_MetaAdapter"                                                        # Only MetaAdapter enabled
#     "--enable_lora --enable_BitFit"                                               # lora and BitFit enabled
#     "--enable_lora --enable_MetaAdapter"                                          # lora and MetaAdapter enabled
#     "--enable_BitFit --enable_MetaAdapter"                                        # BitFit and MetaAdapter enabled
#     "--enable_lora --enable_breaking_loss"                                        # lora and breaking loss
#     "--enable_BitFit --enable_breaking_loss"                                      # BitFit and breaking loss
#     "--enable_MetaAdapter --enable_breaking_loss"                                 # MetaAdapter and breaking loss
#     "--enable_lora --enable_BitFit --enable_MetaAdapter"                          # lora, BitFit and MetaAdapter enabled
#     "--enable_lora --enable_BitFit --enable_breaking_loss"                        # lora, BitFit and breaking loss
#     "--enable_lora --enable_MetaAdapter --enable_breaking_loss"                   # lora, MetaAdapter and breaking loss
#     "--enable_BitFit --enable_MetaAdapter --enable_breaking_loss"                 # BitFit, MetaAdapter and breaking loss
#     "--enable_lora --enable_BitFit --enable_MetaAdapter --enable_breaking_loss"   # lora, BitFit, MetaAdapter and breaking loss
# )

# combinations without breaking loss (for eurosat)
# ENABLE_COMBINATIONS=(
#     "--enable_lora"                                           # Only lora enabled
#     "--enable_BitFit"                                         # Only BitFit enabled
#     "--enable_MetaAdapter"                                    # Only MetaAdapter enabled
#     "--enable_lora --enable_BitFit"                           # Both lora and BitFit enabled
#     "--enable_lora --enable_MetaAdapter"                      # Both lora and MetaAdapter enabled
#     "--enable_BitFit --enable_MetaAdapter"                    # Both BitFit and MetaAdapter enabled
#     "--enable_lora --enable_BitFit --enable_MetaAdapter"      # enable all
# )



# # for eurosat, subset of general combinations that avoid the breaking loss
ENABLE_COMBINATIONS=(
    "--enable_MetaAdapter"
    "--enable_lora --enable_BitFit --enable_MetaAdapter"
    "--enable_lora --enable_MetaAdapter"
    "--enable_BitFit --enable_MetaAdapter"
)

# Loop through datasets, seeds, and enable combinations
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Construct save path
        SAVE_PATH="ckpt/seed_${SEED}"
        if [ "$DATASET" == "circuits" ]; then
            SHOTS_VALUES=(4 8 16)
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
                if [[ "$ENABLE_FLAGS" == *"--enable_MetaAdapter"* ]]; then
                    FILENAME+="_metaAdapter"
                fi
                if [[ "$ENABLE_FLAGS" == *"--enable_breaking_loss"* ]]; then
                    FILENAME+="_breaking"
                fi
                FILENAME+="_${DATASET}_${SHOTS}shots"
                
                # Construct and execute the command
                CMD="${BASE_CMD} --dataset $DATASET --seed $SEED --shots $SHOTS $ENABLE_FLAGS --save_path $SAVE_PATH --filename $FILENAME"
                $CMD
            done
        done
    done
done