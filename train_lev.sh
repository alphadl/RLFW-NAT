#! /usr/bin/bash
ps aux|grep /root/miniconda2/envs/py3.7/bin/python|awk '{print $2}'|xargs kill -9

pip install -e ./fairseq_lev/

set -e

s=$SRC
t=$TGT

task1=raw_PT
task2=bidirectional_KD
task3=forward_KD

task=lev_${task1}_${task2}_${task3}

if [ ! -d ./checkpoint/${s}${t}/${task1} ]; then
  mkdir -p ./checkpoint/${s}${t}/${task1}
fi
if [ ! -d ./checkpoint/${s}${t}/${task2} ]; then
  mkdir -p ./checkpoint/${s}${t}/${task2}
fi
if [ ! -d ./checkpoint/${s}${t}/${task} ]; then
  mkdir -p ./checkpoint/${s}${t}/${task}
fi

echo ">>> training"

#raw pretraining: train 0-2w steps
python ./fairseq_lev/fairseq/fairseq_cli/train.py $TASK1_path \
   --save-dir ./checkpoint/${s}${t}/${task1} \
   --ddp-backend=no_c10d --fp16 \
   --task translation_lev \
   --criterion nat_loss \
   --arch levenshtein_transformer \
   --label-smoothing 0.1 \
   --attention-dropout 0.0 \
   --activation-dropout 0.0 \
   --dropout 0.2 \
   --noise random_delete \
   --share-decoder-input-output-embed \
   --optimizer adam --adam-betas '(0.9,0.98)' \
   --lr 1e-07 --max-lr 1e-3 --lr-scheduler cosine \
   --warmup-init-lr 1e-07 --warmup-updates 10000 --lr-shrink 1 --lr-period-updates 10000 \
   --max-update 20000 \
   --weight-decay 0.0 --clip-norm 0.1 \
   --max-tokens 20000 --update-freq 3 \
   --decoder-learned-pos \
   --encoder-learned-pos \
   --apply-bert-init \
   --no-progress-bar --log-format 'simple' --log-interval 100 \
   --fixed-validation-seed 7 \
   --seed 1 \
   --save-interval-updates 2000 \
   --keep-last-epochs 0 \
   --fp16-scale-tolerance 0.1 > ./checkpoint/${s}${t}/${task1}/train_${s}${t}_${task}_sub_${task1}.log 2>&1

if [ -f ./checkpoint/${s}${t}/${task1}/checkpoint_*_20000.pt ]; then
  cp ./checkpoint/${s}${t}/${task1}/checkpoint_*_20000.pt ./checkpoint/${s}${t}/${task2}/${task1}_20000.pt
  cp ./checkpoint/${s}${t}/${task2}/${task1}_20000.pt ./checkpoint/${s}${t}/${task2}/checkpoint_last.pt
fi

#bidirectional distillation: train 2-4w steps
python ./fairseq_lev/fairseq/fairseq_cli/train.py $TASK2_path \
   --save-dir ./checkpoint/${s}${t}/${task2} \
   --ddp-backend=no_c10d --fp16 \
   --task translation_lev \
   --criterion nat_loss \
   --arch levenshtein_transformer \
   --label-smoothing 0.1 \
   --attention-dropout 0.0 \
   --activation-dropout 0.0 \
   --dropout 0.2 \
   --noise random_delete \
   --share-decoder-input-output-embed \
   --optimizer adam --adam-betas '(0.9,0.98)' \
   --lr 1e-07 --max-lr 1e-3 --lr-scheduler cosine \
   --warmup-init-lr 1e-07 --warmup-updates 10000 --lr-shrink 1 --lr-period-updates 30000 \
   --max-update 40000 \
   --weight-decay 0.0 --clip-norm 0.1 \
   --max-tokens 20000 --update-freq 3 \
   --decoder-learned-pos \
   --encoder-learned-pos \
   --apply-bert-init \
   --no-progress-bar --log-format 'simple' --log-interval 100 \
   --fixed-validation-seed 7 \
   --seed 1 \
   --save-interval-updates 2000 \
   --keep-last-epochs 0 \
   --fp16-scale-tolerance 0.1 > ./checkpoint/${s}${t}/${task2}/train_${s}${t}_${task}_sub_${task2}.log 2>&1
wait

if [ -f ./checkpoint/${s}${t}/${task2}/checkpoint_*_40000.pt ]; then
  cp ./checkpoint/${s}${t}/${task2}/checkpoint_*_40000.pt ./checkpoint/${s}${t}/${task}/${task2}_20000.pt
  cp ./checkpoint/${s}${t}/${task}/${task2}_20000.pt ./checkpoint/${s}${t}/${task}/checkpoint_last.pt
fi

#forward distillation: train 4-7w steps
python ./fairseq_lev/fairseq/fairseq_cli/train.py $TASK3_path \
   --save-dir ./checkpoint/${s}${t}/${task} \
   --ddp-backend=no_c10d --fp16 \
   --task translation_lev \
   --criterion nat_loss \
   --arch levenshtein_transformer \
   --label-smoothing 0.1 \
   --attention-dropout 0.0 \
   --activation-dropout 0.0 \
   --dropout 0.2 \
   --noise random_delete \
   --share-decoder-input-output-embed \
   --optimizer adam --adam-betas '(0.9,0.98)' \
   --lr 1e-07 --max-lr 1e-3 --lr-scheduler cosine \
   --warmup-init-lr 1e-07 --warmup-updates 10000 --lr-shrink 1 --lr-period-updates 60000 \
   --max-update 70000 \
   --weight-decay 0.0 --clip-norm 0.1 \
   --max-tokens 20000 --update-freq 3 \
   --decoder-learned-pos \
   --encoder-learned-pos \
   --apply-bert-init \
   --no-progress-bar --log-format 'simple' --log-interval 100 \
   --fixed-validation-seed 7 \
   --seed 1 \
   --save-interval-updates 2000 \
   --keep-last-epochs 0 \
   --fp16-scale-tolerance 0.1 > ./checkpoint/${s}${t}/${task}/train_${s}${t}_${task}_sub_${task}.log 2>&1
wait

#--reset-lr-scheduler --reset-optimizer --reset-dataloader

# for small dataset e.g. en-ro, we just change following settings:
#   --attention-dropout 0.3 \
#   --activation-dropout 0.3 \
#   --dropout 0.3 \
#   --share-all-embeddings \
#   --warmup-updates 4000--lr-period-updates 21000 \
#   --max-update 25000 \
#   --weight-decay 0.0001