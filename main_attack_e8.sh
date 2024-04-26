# ep8
# model: vgg
# python attack_eval.py --model_name vgg_16 --max_epsilon 8 --attack_method Base --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3_e8/MIM/ --output_file adv_vgg_c3_e8.csv
# python attack_eval.py --model_name vgg_16 --max_epsilon 8 --attack_method FDA --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3_e8/FDA/ --output_file adv_vgg_c3_e8.csv
# python attack_eval.py --model_name vgg_16 --max_epsilon 8 --attack_method NRDM --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3_e8/NRDM/ --output_file adv_vgg_c3_e8.csv
# python attack_eval.py --model_name vgg_16 --max_epsilon 8 --attack_method FIA --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3_e8/FIA/ --output_file adv_vgg_c3_e8.csv
# python attack_eval.py --model_name vgg_16 --max_epsilon 8 --attack_method NAA_P --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3_e8/NAA_P/ --output_file adv_vgg_c3_e8.csv
python attack_eval.py --model_name vgg_16 --max_epsilon 8 --attack_method LID_NM --layer_name vgg_16/conv3/conv3_3/Relu --output_dir adv_vgg_c3_e8/LID_NM/ --output_file adv_vgg_c3_e8.csv

# model: inc3
python attack_eval.py --model_name inception_v3 --max_epsilon 8 --attack_method Base --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b_e8/MIM/ --output_file adv_inc3_m5b_e8.csv 
python attack_eval.py --model_name inception_v3 --max_epsilon 8 --attack_method FDA --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b_e8/FDA/ --output_file adv_inc3_m5b_e8.csv 
python attack_eval.py --model_name inception_v3 --max_epsilon 8 --attack_method NRDM --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b_e8/NRDM/ --output_file adv_inc3_m5b_e8.csv 
python attack_eval.py --model_name inception_v3 --max_epsilon 8 --attack_method FIA --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b_e8/FIA/ --output_file adv_inc3_m5b_e8.csv 
python attack_eval.py --model_name inception_v3 --max_epsilon 8 --attack_method NAA_P --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b_e8/NAA_P/ --output_file adv_inc3_m5b_e8.csv 
python attack_eval.py --model_name inception_v3 --max_epsilon 8 --attack_method LID_NM --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir adv_inc3_m5b_e8/LID_NM/ --output_file adv_inc3_m5b_e8.csv

# model: res1152
python attack_eval.py --model_name resnet_v1_152 --max_epsilon 8 --attack_method Base --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8_e8/MIM/ --output_file adv_res1152_b2u8_e8.csv
python attack_eval.py --model_name resnet_v1_152 --max_epsilon 8 --attack_method FDA --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8_e8/FDA/ --output_file adv_res1152_b2u8_e8.csv
python attack_eval.py --model_name resnet_v1_152 --max_epsilon 8 --attack_method NRDM --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8_e8/NRDM/ --output_file adv_res1152_b2u8_e8.csv
python attack_eval.py --model_name resnet_v1_152 --max_epsilon 8 --attack_method FIA --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8_e8/FIA/ --output_file adv_res1152_b2u8_e8.csv
python attack_eval.py --model_name resnet_v1_152 --max_epsilon 8 --attack_method NAA_P --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8_e8/NAA_P/ --output_file adv_res1152_b2u8_e8.csv
python attack_eval.py --model_name resnet_v1_152 --max_epsilon 8 --attack_method LID_NM --layer_name resnet_v1_152/block2/unit_8/bottleneck_v1/add --output_dir adv_res1152_b2u8_e8/LID_NM/ --output_file adv_res1152_b2u8_e8.csv

# model: incres2
python attack_eval.py --model_name inception_resnet_v2 --max_epsilon 8 --attack_method Base --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a_e8/MIM/ --output_file adv_incres2_c4a_e8.csv
python attack_eval.py --model_name inception_resnet_v2 --max_epsilon 8 --attack_method FDA --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a_e8/FDA/ --output_file adv_incres2_c4a_e8.csv
python attack_eval.py --model_name inception_resnet_v2 --max_epsilon 8 --attack_method NRDM --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a_e8/NRDM/ --output_file adv_incres2_c4a_e8.csv
python attack_eval.py --model_name inception_resnet_v2 --max_epsilon 8 --attack_method FIA --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a_e8/FIA/ --output_file adv_incres2_c4a_e8.csv
python attack_eval.py --model_name inception_resnet_v2 --max_epsilon 8 --attack_method NAA_P --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a_e8/NAA_P/ --output_file adv_incres2_c4a_e8.csv
python attack_eval.py --model_name inception_resnet_v2 --max_epsilon 8 --attack_method LID_NM --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu --output_dir adv_incres2_c4a_e8/LID_NM/ --output_file adv_incres2_c4a_e8.csv
