# Define dataset ID list (assuming train1/val1, train2/val2, etc.)
DATASETS=("train_strategyqa_qwen_unc" "train_strategyqa_qwen_conf")
VAL_DATASETS=("val_strategyqa_qwen_unc" "val_strategyqa_qwen_conf")
OUTPUT_DIR=("train_strategyqa_qwen_unc" "train_strategyqa_qwen_conf")

# You can change the following parameters as needed
# Base parameters
BASE_OUTPUT_DIR="../LLaMA-Factory/saves"
LOG_DIR="../LLaMA-Factory/logs"
BASE_YAML_DIR="../LLaMA-Factory/examples/train_lora"
TEMPLATE_YAML="${BASE_YAML_DIR}/qwen_lora_sft.yaml"

mkdir -p "${LOG_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}" 

# Iterate over datasets
for ((i=0; i<${#DATASETS[@]}; i++)); do
    TRAIN_DATASET="${DATASETS[i]}"
    VAL_DATASET="${VAL_DATASETS[i]}"
    OUTPUT_PATH="${BASE_OUTPUT_DIR}/${OUTPUT_DIR[i]}"
    LOG_FILE="${LOG_DIR}/${OUTPUT_DIR[i]}.log"
    TEMP_YAML="${BASE_YAML_DIR}/${OUTPUT_DIR[i]}.yaml"

    echo "Training on dataset: ${TRAIN_DATASET}, Validation on: ${VAL_DATASET}, Output: ${OUTPUT_PATH}"

    # Use sed to replace placeholders in the YAML template
    cp ${TEMPLATE_YAML} ${TEMP_YAML}
    sed -i "s/{{TRAIN_DATASET}}/${TRAIN_DATASET}/g" ${TEMP_YAML}
    sed -i "s/{{VAL_DATASET}}/${VAL_DATASET}/g" ${TEMP_YAML}
    sed -i "s|{{OUTPUT_DIR}}|${OUTPUT_PATH}|g" ${TEMP_YAML}

    # Run training
    llamafactory-cli train \
        ${TEMP_YAML} \
        > "${LOG_FILE}" 2>&1

    # Optional: Clean up temporary YAML file
    rm ${TEMP_YAML}
done