#!/bin/bash
#
# train_source_model.sh
# author: Li Li (lili-0805@ieee.org)
#


source_model=
train_data=
save_model_path=
teacher_model=""
pretrained_model=""

. ./local/parse_options.sh || exit 1;

case "${source_model}" in
    "CVAE"|"ACVAE")
        learning_rate=0.0001
        ;;
    "ChimeraACVAE")
        learning_rate=0.00001
        ;;
esac

python -m mvae_ss.local.train_source_model \
    --source_model "${source_model}" \
    --train_data "${train_data}" \
    --save_model_path "${save_model_path}" \
    --teacher_model "${teacher_model}" \
    --pretrained_model "${pretrained_model}" \
    --epoch 1000 \
    --snapshot 200 \
    --iteration 9 \
    --learning_rate "${learning_rate}"
