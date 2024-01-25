# model: vgg
# python attack_eval.py --model_name vgg_16 --attack_method Base --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/MIM/ --output_file adv_vgg_c3.csv
# python attack_eval.py --model_name vgg_16 --attack_method FDA --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/FDA/ --output_file adv_vgg_c3.csv
# python attack_eval.py --model_name vgg_16 --attack_method NRDM --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/NRDM/ --output_file adv_vgg_c3.csv
# python attack_eval.py --model_name vgg_16 --attack_method FIA --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/FIA/ --output_file adv_vgg_c3.csv
# python attack_eval.py --model_name vgg_16 --attack_method NAA_P --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/NAA_P/ --output_file adv_vgg_c3.csv
# python attack_eval.py --model_name vgg_16 --attack_method LID_NM --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/LID_NM/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method LID_NM --top ag --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/LID_NM_AG/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method LID_NM --top sg --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/LID_NM_SG/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method LID_NM --top z --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/LID_NM_Z/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method LID_NM --gauss_noise 0 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/LID_NM_NoNoise/ --output_file adv_vgg_c3.csv
python attack_eval.py --model_name vgg_16 --attack_method LID_NM --ens 1 --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3/LID_NM_NoIG/ --output_file adv_vgg_c3.csv

# model: inc3
# python attack_eval.py --model_name inception_v3 --attack_method Base --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b/MIM/ --output_file adv_inc3_m5b.csv 
# python attack_eval.py --model_name inception_v3 --attack_method FDA --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b/FDA/ --output_file adv_inc3_m5b.csv 
# python attack_eval.py --model_name inception_v3 --attack_method NRDM --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b/NRDM/ --output_file adv_inc3_m5b.csv 
# python attack_eval.py --model_name inception_v3 --attack_method FIA --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b/FIA/ --output_file adv_inc3_m5b.csv 
# python attack_eval.py --model_name inception_v3 --attack_method NAA_P --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b/NAA_P/ --output_file adv_inc3_m5b.csv 
# python attack_eval.py --model_name inception_v3 --attack_method LID_NM --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b/LID_NM/ --output_file adv_inc3_m5b.csv

# model: res1152
# python attack_eval.py --model_name resnet_v1_152 --attack_method Base --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8/MIM/ --output_file adv_res1152_b2u8.csv
# python attack_eval.py --model_name resnet_v1_152 --attack_method FDA --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8/FDA/ --output_file adv_res1152_b2u8.csv
# python attack_eval.py --model_name resnet_v1_152 --attack_method NRDM --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8/NRDM/ --output_file adv_res1152_b2u8.csv
# python attack_eval.py --model_name resnet_v1_152 --attack_method FIA --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8/FIA/ --output_file adv_res1152_b2u8.csv
# python attack_eval.py --model_name resnet_v1_152 --attack_method NAA_P --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8/NAA_P/ --output_file adv_res1152_b2u8.csv
# python attack_eval.py --model_name resnet_v1_152 --attack_method LID_NM --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8/LID_NM/ --output_file adv_res1152_b2u8.csv

# model: incres2
# python attack_eval.py --model_name inception_resnet_v2 --attack_method Base --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a/MIM/ --output_file adv_incres2_c4a.csv
# python attack_eval.py --model_name inception_resnet_v2 --attack_method FDA --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a/FDA/ --output_file adv_incres2_c4a.csv
# python attack_eval.py --model_name inception_resnet_v2 --attack_method NRDM --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a/NRDM/ --output_file adv_incres2_c4a.csv
# python attack_eval.py --model_name inception_resnet_v2 --attack_method FIA --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a/FIA/ --output_file adv_incres2_c4a.csv
# python attack_eval.py --model_name inception_resnet_v2 --attack_method NAA_P --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a/NAA_P/ --output_file adv_incres2_c4a.csv
# python attack_eval.py --model_name inception_resnet_v2 --attack_method LID_NM --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a/LID_NM/ --output_file adv_incres2_c4a.csv
