# model: vgg
# TIM is bad
python attack_eval.py --model_name vgg_16 --attack_method Base --num_iter 1 --alpha 16. --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/FGSM/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_DIM --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_DIM/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_PIM --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_PIM/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_NM --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_NM/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_TIM --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_TIM/ --output_file adv_vgg_c3.csv
# adv_vgg_c3/FGSM/              ,69.5%,63.9%,58.6%,77.2%,65.7%,72.1%,65.3%,98.8%,95.0%,61.9%,55.2%,57.8%,57.6%,46.3%
# adv_vgg_c3/MIM/               ,88.7%,88.8%,82.4%,93.0%,88.2%,88.4%,85.0%,99.8%,99.6%,81.7%,72.9%,80.5%,77.5%,65.7%
# adv_vgg_c3/MIM_DIM/           ,90.5%,88.5%,82.3%,93.6%,90.2%,90.9%,84.8%,100.0%,99.3%,86.9%,79.1%,84.7%,83.7%,72.9%
# adv_vgg_c3/MIM_PIM/           ,91.5%,91.1%,84.5%,93.0%,90.0%,90.8%,86.5%,100.0%,99.8%,84.1%,75.5%,82.4%,79.2%,68.2%
# adv_vgg_c3/MIM_NM/            ,92.3%,91.6%,86.5%,94.5%,89.9%,91.4%,86.9%,100.0%,99.9%,83.6%,77.0%,82.6%,78.4%,70.2%
# adv_vgg_c3/MIM_TIM/           ,63.4%,60.9%,42.8%,77.8%,72.1%,57.8%,53.7%,99.9%,97.2%,57.6%,48.5%,57.5%,61.3%,48.3%
# adv_vgg_c3/MIM_DIM_TIM/       ,65.9%,62.7%,48.1%,82.1%,78.0%,63.8%,58.0%,99.8%,97.7%,62.8%,56.7%,67.1%,67.8%,54.5%
# adv_vgg_c3/MIM_DIM_PIM/       ,91.8%,91.0%,85.7%,94.1%,92.1%,91.9%,87.0%,100.0%,99.9%,86.6%,82.4%,86.7%,84.3%,73.4%