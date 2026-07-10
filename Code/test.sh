
for ((epoch=250;epoch<=250;epoch+=5))
do 
    echo "==========testing $epoch====================="
    CUDA_VISIBLE_DEVICES=4  python src/test.py \
    --cfg /home/chenghaotong/CLIPSR_ver3/Code/cfg/Birds.yml \
    --pretrained_model_path '/home/chenghaotong/CLIPSR_ver3/saved_models/bird/ours/ours' \
    --state_epoch $epoch \
    --other 4_CUB_Ours_$epoch \
    --scaler 4
done

for ((epoch=200;epoch<=250;epoch+=10))
do 
    echo "==========testing $epoch====================="
    CUDA_VISIBLE_DEVICES=6  python src/test.py \
    --cfg /home/chenghaotong/CLIPSR_ver3/Code/cfg/CelebA.yml \
    --pretrained_model_path '/home/chenghaotong/CLIPSR_ver3/saved_models/cele/net_nf64_normal_cele_256_2025_12_27_02_16_24/x16' \
    --state_epoch $epoch \
    --other x16_CelebA_OURS_$epoch \
    --scaler 16
done

for ((epoch=200;epoch<=250;epoch+=10))
do 
    echo "==========testing $epoch====================="
    CUDA_VISIBLE_DEVICES=5  python src/test.py \
    --cfg /home/chenghaotong/CLIPSR_ver3/Code/cfg/CelebA.yml \
    --pretrained_model_path '/home/chenghaotong/CLIPSR_ver3/saved_models/cele/net_nf64_normal_cele_256_2025_12_27_02_45_32/x8' \
    --state_epoch $epoch \
    --other x8_CelebA_OURS_$epoch \
    --scaler 8
done
