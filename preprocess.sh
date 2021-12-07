#! /usr/bin/bash

# preprocess data
s=en
t=de

##############################################################################
echo ">>> Standard RAW settings:"

echo ">>> binarize the data"

if [ ! -d ./data/${s}${t}_data/databin/raw_PT ]; then
  mkdir -p ./data/${s}${t}_data/databin/raw_PT
  
  nohup python ./fairseq_mask/fairseq_cli/preprocess.py \
    --source-lang ${s} --target-lang ${t} \
    --trainpref ./data/${s}${t}_data/train_raw \
    --validpref ./data/${s}${t}_data/valid --testpref ./data/${s}${t}_data/test \
    --joined-dictionary \
    --destdir ./data/${s}${t}_data/databin/raw_PT/ \
    --workers 64
fi
wait
echo ">>> binarizing finished"

##############################################################################
echo ">>> Standard forward_KD settings:"

echo ">>> binarize the data"

if [ ! -d ./data/${s}${t}_data/databin/forward_KD ]; then
  mkdir -p ./data/${s}${t}_data/databin/forward_KD
  
  nohup python ./fairseq_mask/fairseq_cli/preprocess.py \
    --source-lang ${s} --target-lang ${t} \
    --trainpref ./data/${s}${t}_data/train_kd \
    --validpref ./data/${s}${t}_data/valid --testpref ./data/${s}${t}_data/test \
    --srcdict ./data/${s}${t}_data/databin/raw_PT/dict.${s}.txt \
    --tgtdict ./data/${s}${t}_data/databin/raw_PT/dict.${t}.txt \
    --destdir ./data/${s}${t}_data/databin/forward_KD/ \
    --workers 64
fi
wait
echo ">>> binarizing finished"

##############################################################################
echo ">>> Standard reversed_KD settings:"

echo ">>> binarize the data"

if [ ! -d ./data/${s}${t}_data/databin/reversed_KD ]; then
  mkdir -p ./data/${s}${t}_data/databin/reversed_KD
  
  nohup python ./fairseq_mask/fairseq_cli/preprocess.py \
    --source-lang ${s} --target-lang ${t} \
    --trainpref ./data/${s}${t}_data/train_bt \
    --validpref ./data/${s}${t}_data/valid --testpref ./data/${s}${t}_data/test \
    --srcdict ./data/${s}${t}_data/databin/raw_PT/dict.${s}.txt \
    --tgtdict ./data/${s}${t}_data/databin/raw_PT/dict.${t}.txt \
    --destdir ./data/${s}${t}_data/databin/reversed_KD/ \
    --workers 64
fi
wait
echo ">>> binarizing finished"

##############################################################################
echo ">>> Mix forward and reversed KD settings:"
# mix bt and kd

if [ ! -d ./data/${s}${t}_data/bidirectional_KD ]; then
  mkdir -p ./data/${s}${t}_data/bidirectional_KD
fi

if [ ! -f ./data/${s}${t}_data/bidirectional_KD/train_mix.${s} ] || [ ! -f ./data/${s}${t}_data/bidirectional_KD/train_mix.${t} ]; then
  cat ./data/${s}${t}_data/train_kd.${s} ./data/${s}${t}_data/train_bt.${s} > ./data/${s}${t}_data/bidirectional_KD/train_mix.${s}
  cat ./data/${s}${t}_data/train_kd.${t} ./data/${s}${t}_data/train_bt.${t} > ./data/${s}${t}_data/bidirectional_KD/train_mix.${t}
fi
echo "concatenating finished"

# deduplicate training data
if [ ! -f ./data/${s}${t}_data/bidirectional_KD/train_mix.dedup ]; then
  paste ./data/${s}${t}_data/bidirectional_KD/train_mix.${s} ./data/${s}${t}_data/bidirectional_KD/train_mix.${t} | awk '!x[$0]++' > ./data/${s}${t}_data/bidirectional_KD/train_mix.dedup
fi

echo "keeping $(wc -l ./data/${s}${t}_data/bidirectional_KD/train_mix.dedup) bitext out of $(wc -l ./data/${s}${t}_data/bidirectional_KD/train_mix.${s})"

if [ ! -f ./data/${s}${t}_data/bidirectional_KD/train_mix.dedup.${s} ] || [ ! -f ./data/${s}${t}_data/bidirectional_KD/train_mix.dedup.${t} ]; then 
  cut -f1 ./data/${s}${t}_data/bidirectional_KD/train_mix.dedup > ./data/${s}${t}_data/bidirectional_KD/train_mix.dedup.${s}
  cut -f2 ./data/${s}${t}_data/mix2bidirectional_KD_kd_bt/train_mix.dedup > ./data/${s}${t}_data/bidirectional_KD/train_mix.dedup.${t}
fi
echo "keeping $(wc -l ./data/${s}${t}_data/mix2_kd_bt/train_mix.dedup.${s}) bitext out of $(wc -l ./data/${s}${t}_data/mix2_kd_bt/train_mix.${s})"

echo ">>> binarize the bidirectional_KD data"

if [ ! -d ./data/${s}${t}_data/databin/bidirectional_KD ]; then
  mkdir -p ./data/${s}${t}_data/databin/bidirectional_KD
  
  nohup python ./fairseq_mask/fairseq_cli/preprocess.py \
    --source-lang ${s} --target-lang ${t} \
    --trainpref ./data/${s}${t}_data/bidirectional_KD/train_mix.dedup \
    --validpref ./data/${s}${t}_data/valid --testpref ./data/${s}${t}_data/test \
    --srcdict ./data/${s}${t}_data/databin/raw_PT/dict.${s}.txt \
    --tgtdict ./data/${s}${t}_data/databin/raw_PT/dict.${t}.txt \
    --destdir ./data/${s}${t}_data/databin/bidirectional_KD/ \
    --workers 64
fi
wait
echo ">>> binarizing finished"