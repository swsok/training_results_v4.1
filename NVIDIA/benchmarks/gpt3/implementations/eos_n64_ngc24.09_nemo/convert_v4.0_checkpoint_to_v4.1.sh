#!/bin/bash

# Directory containing the subdirectories to rename
TARGET_DIR=$1

# Array of regex patterns and replacements
declare -A regex_patterns=(
["language_model"]="model"
["encoder"]="decoder"
["post_attention_layernorm.weight"]="layernorm_mlp.layer_norm_weight"
["post_attention_layernorm.bias"]="layernorm_mlp.layer_norm_bias"
["input_layernorm.weight"]="self_attention.layernorm_qkv.layer_norm_weight"
["input_layernorm.bias"]="self_attention.layernorm_qkv.layer_norm_bias"
["mlp.dense_h_to_4h.weight"]="layernorm_mlp.fc1_weight"
["mlp.dense_h_to_4h.bias"]="layernorm_mlp.fc1_bias"
["mlp.dense_4h_to_h.weight"]="layernorm_mlp.fc2_weight"
["mlp.dense_4h_to_h.bias"]="layernorm_mlp.fc2_bias"
["self_attention.query_key_value.weight"]="self_attention.layernorm_qkv.weight"
["self_attention.query_key_value.bias"]="self_attention.layernorm_qkv.bias"
["self_attention.dense.weight"]="self_attention.proj.weight"
["self_attention.dense.bias"]="self_attention.proj.bias"
["fp32_from_fp16"]="fp32_param"

)

# Iterate over each subdirectory in the target directory
for subdir in "$TARGET_DIR"/*/; do
    # Get the base name of the subdirectory
    subdir_name=$(basename "$subdir")

    # Initialize the new name with the original name
    new_name="$subdir_name"

    # Apply each regex pattern and replacement
    for pattern in "${!regex_patterns[@]}"; do
        replacement="${regex_patterns[$pattern]}"
        new_name=$(echo "$new_name" | sed -E "s/$pattern/$replacement/")
    done

    # If the name has changed, rename the subdirectory
    if [[ "$new_name" != "$subdir_name" ]]; then
        mv "$TARGET_DIR/$subdir_name" "$TARGET_DIR/$new_name"
        echo "Renamed: $subdir_name -> $new_name"
    fi
done

# TODO: review what's in the common.pt
python -c "import torch; x = torch.load('${1}/common.pt'); \
           x['optimizer_states'][0]['state']={'step': 0}; \
           x['optimizer_states'][0]['param_groups']=x['optimizer_states'][0]['optimizer']['param_groups'];
           del x['optimizer_states'][0]['optimizer']; \
           x['pytorch-lightning_version'] = '1.7.7' \
           torch.save(x, '${1}/common.pt')"