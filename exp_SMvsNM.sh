# model: vgg
python attack_eval.py --model_name vgg_16 --attack_method Base --alpha 1.6 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_A10/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base --alpha 2.4 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_A15/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base --alpha 3.2 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_A20/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base --alpha 4.0 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_A25/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base --alpha 4.8 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_A30/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base --alpha 5.6 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_A35/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base --alpha 6.4 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_A40/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_NM --alpha 1.6 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_NM_A10/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_NM --alpha 2.4 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_NM_A15/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_NM --alpha 3.2 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_NM_A20/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_NM --alpha 4.0 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_NM_A25/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_NM --alpha 4.8 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_NM_A30/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_NM --alpha 5.6 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_NM_A35/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method Base_NM --alpha 6.4 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM_NM_A40/ --output_file adv_vgg_c3.csv
# adv_vgg_c3/MIM_A10/           ,89.0%,89.0%,82.3%,93.1%,88.3%,88.7%,84.7%,99.9%,99.6%,82.5%,72.1%,79.7%,77.9%,66.1%
# adv_vgg_c3/MIM_A15/           ,90.6%,91.4%,83.2%,93.7%,90.4%,91.0%,86.8%,100.0%,99.7%,85.6%,74.8%,80.7%,79.0%,67.5%
# adv_vgg_c3/MIM_A20/           ,91.0%,91.0%,84.3%,93.4%,89.7%,91.1%,85.9%,100.0%,99.7%,83.7%,74.4%,81.7%,78.1%,67.4%
# adv_vgg_c3/MIM_A25/           ,92.0%,91.3%,85.5%,92.8%,89.7%,91.2%,86.7%,100.0%,99.7%,82.9%,74.0%,79.8%,76.2%,66.2%
# adv_vgg_c3/MIM_A30/           ,91.1%,90.9%,86.0%,93.4%,89.7%,90.9%,87.0%,100.0%,99.8%,83.1%,73.3%,81.1%,75.8%,66.7%
# adv_vgg_c3/MIM_A35/           ,92.3%,91.8%,86.0%,93.2%,89.0%,90.8%,86.8%,100.0%,99.8%,83.0%,72.6%,80.4%,74.0%,64.0%
# adv_vgg_c3/MIM_A40/           ,92.1%,92.2%,86.4%,92.6%,89.5%,90.7%,87.6%,100.0%,99.7%,83.1%,74.1%,79.5%,74.8%,64.9%
# adv_vgg_c3/MIM_NM_A10/        ,87.8%,86.5%,79.1%,90.9%,86.8%,86.4%,81.0%,100.0%,99.7%,80.5%,70.9%,78.0%,76.3%,64.8%
# adv_vgg_c3/MIM_NM_A15/        ,90.2%,90.0%,83.4%,92.7%,88.3%,89.8%,84.2%,99.9%,99.8%,83.8%,76.6%,82.2%,78.1%,68.4%
# adv_vgg_c3/MIM_NM_A20/        ,91.2%,90.9%,84.3%,92.6%,89.2%,90.2%,86.9%,100.0%,99.7%,84.4%,75.5%,82.4%,78.5%,69.6%
# adv_vgg_c3/MIM_NM_A25/        ,91.8%,91.2%,85.5%,93.5%,89.4%,90.9%,87.2%,100.0%,99.7%,85.0%,77.3%,82.1%,78.0%,71.0%
# adv_vgg_c3/MIM_NM_A30/        ,92.7%,91.5%,86.4%,93.9%,89.7%,90.6%,88.1%,100.0%,99.7%,84.4%,77.6%,83.0%,78.8%,71.0%
# adv_vgg_c3/MIM_NM_A35/        ,92.5%,91.9%,86.2%,93.6%,90.4%,91.6%,86.8%,100.0%,99.8%,84.5%,77.3%,81.8%,77.4%,69.1%
# adv_vgg_c3/MIM_NM_A40/        ,92.1%,92.7%,86.1%,94.0%,90.5%,91.5%,87.3%,100.0%,99.8%,84.4%,77.1%,81.0%,77.5%,68.6%
