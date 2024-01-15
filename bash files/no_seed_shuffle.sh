#!/bin/bash

export NVIDIA_VISIBLE_DEVICES=3 \
export CUDA_VISIBLE_DEVICES=3 \



lr=3e-5
cl_temp=0.05
distill_weight=0
distill_temp1=0.025
distill_temp2=0.0125

#values=("61" "62" "63" "64" "65")
#values=("66" "67" "68" "69" "70")
#values=("81" "82" "83" "84" "85")
values=("86" "87" "88" "89" "90")

for exp_num in "${values[@]}"; do

    CL_steps=5000
    distillation_stopping_steps=5000

    train_file=data/shuf_exp$exp_num.txt
    shuf data/wiki1m_for_simcse.txt >> $train_file

    
    distillation_stopping_steps=$distillation_stopping_steps


    distill_teacher=CL_no_seed_shuffle/test_shuf_exp$exp_num-warmup_steps$CL_steps-distillation_stopping_steps$distillation_stopping_steps
    cp -r unsup-simcse-bert-base-uncased $distill_teacher
    model=bert-base-uncased

    #group_size_by_prob=0.1

    output_dir=$distill_teacher
    echo output_dir is $output_dir


    warmup_dir=$output_dir/warmup_ckpt


    python3 train_distill_calibrate.py \
        --distill_teacher $distill_teacher --distill_weight $distill_weight --distill_temp1 $distill_temp1 --distill_temp2 $distill_temp2 \
        --model_name_or_path $model \
        --train_file $train_file \
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

