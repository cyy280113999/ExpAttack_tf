"""Implementation of attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import utils
import tqdm
import argparse


slim = tf.contrib.slim


def generate_attack(args):
    # in order to reuse the generation of adv-samples('generate_attack'), generation define as a funtion.
    # main function accept 'args' that following the 'attack_parser'
    # =================================================== main ================================================================#
    # build model graph
    with tf.Graph().as_default():
        with tf.Session() as sess:
            args.sess = sess  # set global parameters
            # into details
            use_model(args)  # omit two indents

def use_model(args):
    # in order to use variables in tf.graph globally and easily. any functions should defined in the main function
    # ========== global functions ================ #
    avaiable_methods={
        'CE':'CE',  # loss mode
        'P':'P',
        'Z':'Z',
        'Base':'Base',  # 
        'FDA':'FDA',
        'NRDM':'NRDM',
        'FIA':'FIA',
        'NAA':'NAA',
        'LID':'LID',
        'DIM':'DIM',
        'TIM':'TIM',
        # 'MIM':'MIM',  # use default
        'PIM':'PIM',
        'NM':'NM',  # FGNM
        'UW':'UW',  # update weight
        'FM':'FM',  # feature momentum
    }
    def get_model_specific_settings(args):
        args.attack_method = args.attack_method.split('_')  # parse attack method into components
        # for atks in args.attack_method:  # check
        #     assert atks in avaiable_methods.values(), '{} is not available'.format(atks)
        args.image_size = utils.image_size[args.model_name]
        args.num_classes = 1000 + utils.offset[args.model_name]
        args.x_shape = [args.batch_size, args.image_size, args.image_size, 3]
        if args.model_name in ['vgg_16', 'vgg_19', 'resnet_v1_50', 'resnet_v1_152']:
            args.eps = args.max_epsilon
            args.alpha = args.alpha
        else:
            args.eps = 2.0 * args.max_epsilon / 255.0
            args.alpha = args.alpha * 2.0 / 255.0
        args.image_preprocessing_fn = utils.normalization_fn_map[args.model_name]
        args.inv_image_preprocessing_fn = utils.inv_normalization_fn_map[args.model_name]
        return args
    def DIM(args, input_tensor):
        """Input diversity: https://arxiv.org/abs/1803.06978"""
        rnd = tf.random_uniform((), args.image_size, args.image_resize, dtype=tf.int32)
        rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        h_rem = args.image_resize - rnd
        w_rem = args.image_resize - rnd
        pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
        padded.set_shape((input_tensor.shape[0], args.image_resize, args.image_resize, 3))
        ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(args.prob), lambda: padded, lambda: input_tensor)
        ret = tf.image.resize_images(ret, [args.image_size, args.image_size],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return ret

    def TIM_generate_kern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        import scipy.stats as st
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
        stack_kernel = np.expand_dims(stack_kernel, 3)
        return stack_kernel
    T_kern = TIM_generate_kern(args.Tkern_size)
    def TIM(grad):
        grad = tf.nn.depthwise_conv2d(grad, T_kern, strides=[1, 1, 1, 1], padding='SAME')
        return grad
    def norm_graph(x, level):  # return norm of tensor
        if level == 1:
            norm = tf.reduce_sum(tf.abs(x), [1, 2, 3], keep_dims=True)
        elif level == 2:
            sqr_sum = tf.reduce_sum(tf.square(x), [1, 2, 3], keep_dims=True)
            norm = tf.sqrt(sqr_sum)
        return norm
    # def normalize_graph(x, level):  # return normalized tensor
    #     if level == 1:
    #         abs_sum = tf.reduce_sum(tf.abs(x), [1, 2, 3], keep_dims=True)
    #         x = x / (abs_sum + 0)
    #     elif level == 2:
    #         sqr_sum = tf.reduce_sum(tf.square(x), [1, 2, 3], keep_dims=True)
    #         x_norm = tf.sqrt(sqr_sum)
    #         x = x / (x_norm + 0)
    #     return x
    def PIM_project_kern(kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
        stack_kern = np.expand_dims(stack_kern, 3)
        return stack_kern, kern_size // 2
    P_kern, P_kernel_radius = PIM_project_kern(args.Pkern_size)
    def PIM_project_noise(x):
        radius = P_kernel_radius
        x = tf.pad(x, [[0, 0], [radius, radius], [radius, radius], [0, 0]], "CONSTANT")
        x = tf.nn.depthwise_conv2d(x, P_kern, strides=[1, 1, 1, 1], padding='VALID')
        return x
    def PIM_Update(x, g, amplification_update):
        # amplification factor
        alpha_beta = alpha * args.amplification_factor
        gamma = args.gamma * alpha_beta
        # Project cut noise
        amplification_update += alpha_beta * tf.sign(g)
        cut_noise = tf.clip_by_value(abs(amplification_update) - eps_std, 0.0, 10000.0) * tf.sign(amplification_update)
        projection = gamma * tf.sign(PIM_project_noise(cut_noise))
        amplification_update += projection
        x = x + alpha_beta * tf.sign(g) + projection
        return x, amplification_update
    
    args = get_model_specific_settings(args)
    # expose some parameters
    sess = args.sess
    num_classes=args.num_classes
    x_shape = args.x_shape
    batch_size = args.batch_size
    layer_name = args.layer_name
    alpha = args.alpha
    eps_std = args.eps
    image_preprocessing_fn = args.image_preprocessing_fn
    inv_image_preprocessing_fn = args.inv_image_preprocessing_fn
    # ========================================================== build graph ================================================================ #
    # input
    x_ph = tf.placeholder(tf.float32, shape=x_shape)
    label_ph = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
    # trained parameters-loading must be after graph-building
    network_fn = utils.nets_factory.get_network_fn(args.model_name, num_classes=num_classes, is_training=False)
    # output
    # whether using DIM or not
    if 'DIM' in args.attack_method:
        logits, _ = network_fn(DIM(args, x_ph))
    else:
        logits, _ = network_fn(x_ph)
    # after network build once, load trained model 
    saver = tf.train.Saver()
    saver.restore(sess, utils.checkpoint_paths[args.model_name])
    problity = tf.nn.softmax(logits, axis=1)
    pred = tf.argmax(logits, axis=1)
    one_hot = tf.one_hot(pred, num_classes)
    # feature
    def get_feature(layer_name):
        operations = tf.get_default_graph().get_operations()
        for op in operations:
            if layer_name == op.name:
                feature=op.outputs[0]
                shape = op.outputs[0].shape
                return feature, shape
        raise Exception('layer name is not correct')
    feature, feat_shape = get_feature(layer_name)
    args.feat_shape = feat_shape
    feature_weight_ph = tf.placeholder(dtype=tf.float32, shape=feat_shape)
    base_feature_ph = tf.placeholder(dtype=tf.float32, shape=feat_shape)
    delta_feat = feature - base_feature_ph
    # loss 
    ce_pred = tf.losses.softmax_cross_entropy(one_hot, logits)  # when to use the labels of prediction?
    ce_gt = tf.losses.softmax_cross_entropy(label_ph, logits)
    logit_loss = tf.reduce_sum(logits*label_ph)
    prob_loss = tf.reduce_sum(problity*label_ph)
    # --------------------------------- attack loss -------------------------------------- #
    attack_loss = None
    # pre-define some feature-level loss. to generate feature gradient
    # options:['CE', 'P', 'Z']
    if 'CE' in args.attack_method:
        feat_loss = -ce_gt  # note that it is negative ce loss
    elif 'P' in args.attack_method:
        feat_loss = prob_loss
    else: # 'Z': logits
        feat_loss = logit_loss
    def FDA_loss():
        mean_tensor = tf.stack([tf.reduce_mean(base_feature_ph, -1), ] * base_feature_ph.shape[-1], -1)  # use clean feature
        small_gate = tf.to_float(base_feature_ph < mean_tensor)  # gate 
        big_gate = tf.to_float(base_feature_ph >= mean_tensor)
        return -tf.log(tf.nn.l2_loss(small_gate * feature)) + tf.log(tf.nn.l2_loss(big_gate * feature))  # enhance small feature, supress big feature
    def NRDM_loss():
        return -tf.norm(feature - base_feature_ph)  # adv - clean
    # to use feature activation * gradient as attack loss
    def WA_loss(feature, weight=None):
        if weight is not None:
            feature = feature * weight
        loss = tf.reduce_sum(feature)
        return loss
    # -----------normal method
    if 'Base' == args.attack_method[0]:
        attack_loss = feat_loss  # same loss
    # --------feature level loss
    elif 'FDA' == args.attack_method[0]:  # attack method 0 is the main idea of attacking
        attack_loss = FDA_loss()
    elif 'NRDM' == args.attack_method[0]:
        attack_loss = NRDM_loss()
    # > feature-level attack generally contains two graphs: attack graph and feature graph.
    # attack graph commonly use weighted-activation loss as attack-loss to obtain gradient of input and to update input
    # feature graph aims to obtain feature-level details, such as activation and gradient of feature
    elif 'FIA' == args.attack_method[0]:
        # build FIA graph
        feat_grad = tf.gradients(feat_loss, feature)[0]
        # run smooth feature grad in session... set weights_ph
        attack_loss = WA_loss(feature, feature_weight_ph)  # the difference between methods is only the weight of feature. 
    elif 'NAA' == args.attack_method[0]:
        feat_grad = tf.gradients(feat_loss, feature)[0]
        attack_loss = WA_loss(delta_feat, feature_weight_ph)
    elif 'Taylor' == args.attack_method[0]:  # taylor is simplified FIA
        feat_grad = tf.gradients(feat_loss, feature)[0]
        attack_loss = WA_loss(feature, feature_weight_ph)  # if you do well in calculus, you know it is nothing else when 'feature_weight_ph=feat_grad'
    elif 'SGM' == args.attack_method[0]:  # smooth grad method
        feat_grad = tf.gradients(feat_loss, feature)[0]
        attack_loss = WA_loss(feature, feature_weight_ph)
    elif 'LID' == args.attack_method[0]:
        softmax_gradient = tf.gradients(tf.reduce_sum(problity*label_ph), logits)[0]
        # feed sig logits weight
        SIG_weight_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_classes])
        SIG_Loss=tf.reduce_sum(logits * SIG_weight_ph)
        feat_grad = tf.gradients(SIG_Loss, feature)[0]
        # run sig feature grad
        attack_loss = WA_loss(delta_feat, feature_weight_ph)
    else:
        raise Exception(f'{args.attack_method} is not available.')
    # -------------------------------------- attack update ----------------------------------------- #
    attack_loss = -attack_loss # optim object from minimize loss to maximize loss. why? PIM code.
    assert attack_loss is not None
    grad = tf.gradients(attack_loss, x_ph)[0]
    # whether using TIM or not
    if 'TIM' in args.attack_method:
        grad = TIM(grad)
    # whether using MIM(optimization process with momentum) 
    if 'MIM' in args.attack_method:
        use_MIM=True
    # use by default 
    use_MIM=True 
    if use_MIM:
        def MomentumGrad(grad, last_grad, keep=1.0):
            grad = grad + keep * last_grad
            return grad
        accumulated_grad_ph = tf.placeholder(dtype=tf.float32, shape=x_shape)
        grad = grad / norm_graph(grad, 1)
        grad = MomentumGrad(grad, accumulated_grad_ph, args.momentum)
    # --update
    adv_input_update = x_ph
    # whether using PIM or not
    amplification_ph = tf.placeholder(dtype=tf.float32, shape=x_shape)
    amplification_update = amplification_ph
    if 'PIM' in args.attack_method:
        adv_input_update, amplification_update = PIM_Update(adv_input_update, grad, amplification_update)
    elif 'NM' in args.attack_method:  # FGNM. amplifing alpha to '3.0' times as same alpha used.
        adv_input_update = adv_input_update + args.nonsign_scale * alpha * norm_graph(tf.sign(grad), 2)/ norm_graph(grad, 2) * grad
    else:  # FGSM
        adv_input_update = adv_input_update + alpha * tf.sign(grad)  # this code maximize loss

    # note : tf graph use too many global variables. better define other functions in the main function.
    def normalize(grad, opt=2):
        if opt == 0:
            nor_grad = grad
        elif opt == 1:
            abs_sum = np.sum(np.abs(grad), axis=(1, 2, 3), keepdims=True)
            if np.any(abs_sum == 0):
                abs_sum[abs_sum == 0] = 1
            nor_grad = grad / abs_sum
        elif opt == 2:
            square = np.sum(np.square(grad), axis=(1, 2, 3), keepdims=True)
            if np.any(square == 0):
                square[square==0] = 1
            nor_grad = grad / np.sqrt(square)
        return nor_grad

    def FIA_weight(cleans_std, labels):
        weight_np = np.zeros(shape=feat_shape)
        for l in range(args.ens):
            mask = np.random.binomial(1, args.probb, size=args.x_shape)
            masked_x = cleans_std * mask  # mask ~std works same as mask ~raw
            w = sess.run(feat_grad, feed_dict={x_ph: masked_x, label_ph: labels})
            weight_np = weight_np + w
        weight_np = normalize(weight_np, 2)
        return weight_np
    def SGM_weight(cleans_std, labels, sigma_scale=1.0):
        weight_np = np.zeros(shape=feat_shape)
        for l in range(args.ens):
            noised = cleans_std + np.random.normal(size=cleans_std.shape, loc=0.0, scale=sigma_scale * eps_std)
            w = sess.run(feat_grad, feed_dict={x_ph: noised, label_ph: labels})
            weight_np = weight_np + w
        weight_np = normalize(weight_np, 2)
        return weight_np
    def NAA_weight(cleans_std, labels):
        weight_np = np.zeros(shape=feat_shape)
        for l in range(args.ens):
            ratio = (l+1)/args.ens
            x_base = image_preprocessing_fn(np.zeros_like(cleans_std))
            ratio_x = cleans_std * ratio + x_base * (1-ratio)
            ratio_x += np.random.normal(size=cleans_std.shape, loc=0.0, scale=0.2)  # Why NAA add noise?
            w = sess.run(feat_grad, feed_dict={x_ph: ratio_x, label_ph: labels})
            weight_np = weight_np + w
        weight_np = normalize(weight_np, 2)
        return weight_np
    def LID_weight(cleans_std, labels):
        sigma_scale = args.gauss_noise
        cleans = inv_image_preprocessing_fn(cleans_std)
        # # def one_hot_encode(labels, num_classes):
        # #     one_hot_labels = np.zeros((len(labels), num_classes))
        # #     one_hot_labels[np.arange(len(labels)), labels] = 1
        # #     return one_hot_labels
        # def softmax(z):
        #     return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        # def softmaxgradient(z, labels): # softmax gradient
        #     p = softmax(z)
        #     return (labels-p) * p
        # def average_softmaxgradient(z, label):
        #     ag = np.zeros(shape=[batch_size, num_classes])
        #     n=10
        #     for l in range(n):
        #         ratio = (l+1)/n
        #         ratio_z = ratio*z
        #         ag+=softmaxgradient(ratio_z, labels)
        #     return ag/n
        # z = sess.run(logits, feed_dict={x_ph: cleans_std, label_ph: labels})
        sig_weight = np.zeros_like(labels)
        if args.top == 'ag':
            n=10
            for l in range(10):
                ratio = (l+1)/args.ens
                ratio_x = cleans * ratio
                ratio_x = image_preprocessing_fn(ratio_x)
                sig_weight += sess.run(softmax_gradient, feed_dict={x_ph: ratio_x, label_ph: labels})
            sig_weight/=n    # average over softmax gradient
        elif args.top == 'sg':
            sig_weight = sess.run(softmax_gradient, feed_dict={x_ph: cleans_std, label_ph: labels})  # softmax gradient
        else:  # args.top == 'z':
            sig_weight = labels  # one-hot
        feat_weight = np.zeros(shape=feat_shape)
        for l in range(args.ens):
            ratio = (l+1)/args.ens
            ratio_x = cleans * ratio
            ratio_x = image_preprocessing_fn(ratio_x)
            ratio_x += np.random.normal(size=x_shape, loc=0.0, scale=sigma_scale * eps_std)
            w = sess.run(feat_grad, feed_dict={x_ph: ratio_x, label_ph: labels, SIG_weight_ph: sig_weight})
            feat_weight = feat_weight + w
        feat_weight = normalize(feat_weight, 2)
        if args.weight_trans == 'square':
            feat_weight = np.square(feat_weight) * np.sign(feat_weight)
        return feat_weight
    
    # ==================================================================== run ============================================================ #
    def preprocess_label(args, labels):
        if args.model_name in ['resnet_v1_50', 'resnet_v1_152', 'vgg_16', 'vgg_19']:
            labels = labels - 1
        # obtain true label
        labels = to_categorical(np.concatenate([labels], axis=-1), args.num_classes)
        # labels = sess.run(one_hot, feed_dict={ori_input: images_tmp, adv_input: images_tmp})
        return labels

    for cleans, names, labels in tqdm.tqdm(utils.load_image(args.input_dir, args.image_size, args.batch_size), desc=args.output_dir):
        # cleans in [0,255]
        cleans_std = image_preprocessing_fn(cleans)  # standardized, in [?, ?]
        # adv = np.copy(cleans)
        adv_std = np.copy(cleans_std)  # notice! standardized
        labels = preprocess_label(args, labels)
        grad_np = np.zeros(shape=x_shape)
        amplification = np.zeros(shape=x_shape)
        feature_weight = np.zeros(shape=feat_shape)  # weighted activation loss
        last_feat_weight = np.zeros(shape=feat_shape)  # feature momentum
        # base
        if 'FDA' in args.attack_method:
            base_feature = sess.run(feature, feed_dict={x_ph: cleans_std})  # base clean
        elif 'NRDM' in args.attack_method:
            base_feature = sess.run(feature, feed_dict={x_ph: cleans_std})  # base clean
        else:
            zeros_images = image_preprocessing_fn(np.zeros_like(cleans))  # base zero
            base_feature = sess.run(feature, feed_dict={x_ph: zeros_images})
        if 'NRDM' in args.attack_method:
            # add some noise to avoid F_{k}(x)-F_{k}(x')=0
            adv_std += np.random.normal(0, 0.1, size=x_shape)
        # optimization
        for i in range(args.num_iter):
            # init
            # ['UW'] controls to update weight in iterations
            if i==0 or 'UW' in args.attack_method:
                # froze weight
                if 'FIA' in args.attack_method:
                    feature_weight = FIA_weight(adv_std, labels)
                elif 'NAA' in args.attack_method:
                    feature_weight = NAA_weight(adv_std, labels)
                elif 'LID' in args.attack_method:
                    feature_weight = LID_weight(adv_std, labels)
                elif 'Taylor' in args.attack_method:
                    feature_weight = sess.run(feat_grad, feed_dict={x_ph: adv_std, label_ph: labels})
                elif 'SGM' in args.attack_method:
                    feature_weight = SGM_weight(adv_std, labels)
                else:
                    pass # no weight
                if 'FM' in args.attack_method:
                    feature_weight = last_feat_weight* args.feat_mom + feature_weight
                last_feat_weight = feature_weight
            # update
            adv_std, grad_np, amplification = sess.run([adv_input_update, grad, amplification_update],
                                                          feed_dict={x_ph: adv_std, label_ph: labels,
                                                                    feature_weight_ph: feature_weight,
                                                                    base_feature_ph: base_feature,
                                                                    accumulated_grad_ph: grad_np,
                                                                    amplification_ph: amplification})
            adv_std = np.clip(adv_std, cleans_std - eps_std, cleans_std + eps_std)  # eps is after std
        adv = inv_image_preprocessing_fn(adv_std)  # unstandardize
        utils.save_image(adv, names, args.output_dir)
    print(f'save {args.output_dir}')

def attack_parser():
    parser = argparse.ArgumentParser(description='Adversarial Attack Parameters')
    # key param
    parser.add_argument('--attack_method', type=str, default='NAA', help='The name of attack method.')
    # common parameters
    parser.add_argument('--model_name', type=str, default='inception_v3', help='The Model used to generate adv.')
    parser.add_argument('--layer_name', type=str, default='InceptionV3/InceptionV3/Mixed_5b/concat', help='The layer to be attacked.')
    parser.add_argument('--input_dir', type=str, default='./dataset/images/', help='Input directory with images.')
    parser.add_argument('--output_dir', type=str, default='./adv/NAA_inc3/', help='Output directory with images.')
    parser.add_argument('--max_epsilon', type=float, default=16.0, help='Maximum size of adversarial perturbation.')
    parser.add_argument('--num_iter', type=int, default=10, help='Number of iterations.')
    parser.add_argument('--alpha', type=float, default=1.6, help='Step size.')
    parser.add_argument('--batch_size', type=int, default=8, help='How many images process at one time.')
    parser.add_argument('--momentum', type=float, default=1.0, help='Momentum.')
    # specific param
    # Parameters for DIM
    parser.add_argument('--image_resize', type=int, default=331, help='Size of each diverse images.')
    parser.add_argument('--prob', type=float, default=0.7, help='Probability of using diverse inputs.')

    # Parameters for TIM
    parser.add_argument('--Tkern_size', type=int, default=15, help='Kernel size of TIM.')

    # Parameters for FIA
    parser.add_argument('--probb', type=float, default=0.9, help='Keep probability = 1 - drop probability.')
    # Parameters for NAA
    parser.add_argument('--ens', type=int, default=30, help='Aggregated N for NAA or Mask number for FIA.')
    # LID
    parser.add_argument('--top', type=str, default='ag', help='whether use sig or not. pass ag to use sig init, see codes for other details.')
    parser.add_argument('--gauss_noise', type=float, default=2.0, help='Guassian noise in integral path.')
    parser.add_argument('--nonsign_scale', type=float, default=3.0, help='Guassian noise in integral path.')
    parser.add_argument('--weight_trans', type=str, default='no', help='weight transform function')
    
    # FMAA
    parser.add_argument('--feat_mom', type=float, default=1.1, help='Feature Momentum.')
    
    # Parameters for PIM
    parser.add_argument('--amplification_factor', type=float, default=2.5, help='To amplify the step size.')
    parser.add_argument('--gamma', type=float, default=0.5, help='The gamma parameter.')
    parser.add_argument('--Pkern_size', type=int, default=3, help='Kernel size of PIM.')
    
    return parser  # add other params

def generate_experiment_name(parameters):
    experiment_name_parts = []
    for key, value in parameters.items():
        # 将键和值转为字符串，并添加到列表中
        key_str = str(key)
        value_str = str(value)
        experiment_name_parts.append(f"{key_str}_{value_str}")

    # 将列表中的字符串连接起来，并用下划线分隔
    experiment_name = "_".join(experiment_name_parts)
    return experiment_name


if __name__ == '__main__':
    parser = attack_parser()
    args = parser.parse_args()
    generate_attack(args)
    
