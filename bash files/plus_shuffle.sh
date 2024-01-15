#!/bin/bash

export NVIDIA_VISIBLE_DEVICES=3 \
export CUDA_VISIBLE_DEVICES=3 \

lr=3e-5
cl_temp=0.05
distill_weight=0
distill_temp1=0.025
distill_temp2=0.0125


seed=50
values=("0.1" "0.05" "0.15" "0.2")
CL_steps=5500  
distillation_stopping_steps=11000

for group_size_by_prob in "${values[@]}"; do
    seed=$seed
    CL_steps=$CL_steps
    distillation_stopping_steps=$distillation_stopping_steps


    distill_teacher=plus_shuffle/Jan6prob$group_size_by_prob-seed$seed-warmup_steps$CL_steps-distillation_stopping_steps$distillation_stopping_steps
    cp -r unsup-simcse-bert-base-uncased $distill_teacher
    model=bert-base-uncased

    output_dir=$distill_teacher
    echo output_dir is $output_dir


    warmup_dir=$output_dir/warmup_ckpt


    python3 train_distill_calibrate.py \
        --group_shuffling --group_size_by_prob $group_size_by_prob \
        --distill_teacher $distill_teacher --distill_weight $distill_weight --distill_temp1 $distill_temp1 --distill_temp2 $distill_temp2 \
        --model_name_or_path $model \
        --train_file data/wiki1m_for_simcse.txt \
        --output_dir $output_dir \
        --num_train_epochs 1 \
        --per_device_train_batch_size 64 \
        --learning_rate $lr \
        --max_seq_length 32 \
        --evaluation_strategy steps \
        --metric_for_best_model stsb_spearman \
        --load_best_model_at_end \
        --eval_steps 125 \
        --pooler_type cls \
        --mlp_only_train \
        --overwrite_output_dir \
        --temp $cl_temp \
        --do_train \
        --seed $seed \
        --stopping_steps $distillation_stopping_steps \
        --CL_steps $CL_steps

    rm -rf $warmup_dir/warmup_ckpt

    echo CL test eval >> $output_dir/eval.txt
    python3 evaluation.py --model_name_or_path $warmup_dir --pooler cls_before_pooler >> $output_dir/eval.txt
    cat $output_dir/eval.txt

    echo OTF distill eval >> $output_dir/eval.txt
    python3 evaluation.py --model_name_or_path $output_dir --pooler cls_before_pooler >> $output_dir/eval.txt
    cat $output_dir/eval.txt

done

