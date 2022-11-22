#!/bin/bash
#
# run.sh:
#   main file to run experiments
# author: Li Li (lili-0805@ieee.org)
#


# stage settings
# 0: train source model
# 1: test (pre-)trained model
stage=0
stop_stage=1

# algorithm: "MVAE" or "FastMVAE" or "FastMVAE2"
algorithm="FastMVAE2"

# dataset: "vcc" or "wsj0" or your own dataset
# note that "wsj0" dataset is not provided in this repository
dataset="vcc"
# set corresponding label dimension
label_dim=

# input and output roots
data_root="../data/"
model_root="../model/"
output_root="../output/"

alpha=0
label_type="cont"
teacher_model=

# test setting
# test mode:
#   "pretrained": test pretrained model in $HOME/model/pretrained_model/
#   "trained": test trained model in stage 0
test_mode="trained"
# dataset for test:
#   "test_input": provided test data
test_dataset="test_input"

. ./local/parse_options.sh || exit 1;

#   default settings for different datasets and algorithms
if [[ "${dataset}" == "vcc" ]]; then
    label_dim=4
elif [[ "${dataset}" == "wsj0" ]]; then
    label_dim=101
fi

# alpha and label_type are parameters for PoE estimation of FastMVAE
# the default teacher model used for training ChimeraACVAE is the provided
#   CVAE source model.
if [[ "${algorithm}" == "MVAE" ]]; then
    source_model="CVAE"
elif [[ "${algorithm}" == "FastMVAE" ]]; then
    source_model="ACVAE"
elif [[ "${algorithm}" == "FastMVAE2" ]]; then
    source_model="ChimeraACVAE"
    teacher_model="${model_root}pretrained_model/CVAE_${dataset}.model"
else
    echo "algorithm is invalid!"
    exit 1
fi

# input directories
train_data="${data_root}${dataset}/"
# output directories
save_model_root="${model_root}trained/${dataset}/${source_model}/"
save_output_root="${output_root}${test_mode}/${algorithm}/train_with_${dataset}/"

# stage 0: train source model
if [[ "${stage}" -le 0 ]] && [[ "${stop_stage}" -ge 0 ]]; then
    echo "stage 0: start to train ${source_model} with ${dataset} dataset..."
    mkdir -p "${save_model_root}"
    ./local/train_source_model.sh \
        --source_model "${source_model}" \
        --train_data "${train_data}" \
        --save_model_path "${save_model_root}" \
        --teacher_model "${teacher_model}"
    echo "stage 0: ${source_model} training is done! Model is saved in ${save_model_root}."
fi

# stage 1: test (pre-)trained model
if [[ "${stage}" -le 1 ]] && [[ "${stop_stage}" -ge 1 ]]; then
    if [[ "${test_mode}" == "pretrained" ]]; then
        model_path="${model_root}pretrained_model/${source_model}_${dataset}.model"
    elif [[ "${test_mode}" == "trained" ]]; then
        model_path="${save_model_root}1000.model"
    fi

    echo "stage 1: start to test the ${test_mode} ${source_model} model of ${model_path}..."

    test_data="${data_root}${test_dataset}/"
    output_path="${save_output_root}${test_dataset}/"
    if [[ "${algorithm}" == "FastMVAE" ]]; then
        output_path+="type_${label_type}_alpha_${alpha}/"
    fi
    mkdir -p $output_path
    echo "  start to test data in ${test_data} and save output in ${output_path}..."
    python -m mvae_ss.local.separation \
        --algorithm "${algorithm}" \
        --source_model "${source_model}" \
        --input_dir "${test_data}" \
        --output_dir "${output_path}" \
        --label_dim "${label_dim}" \
        --model_path "${model_path}" \
        --label_type "${label_type}" \
        --alpha "${alpha}"
    echo "stage 1: testing ${test_mode} ${source_model} model is done!"
fi
