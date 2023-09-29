mmaction2_models_list = [
    'slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb',
    'slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb',
    'slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb',
    'slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb',
    'slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb',
    'slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb',
    'slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb',
    'slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb',
    'slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb',
    'slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb',
    'slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb',
    'slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb',
    'slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb',
    'slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb',
    'slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb',
    'slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb',
    'slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb',
    'vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb',
    'vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb',
    'c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb',
    'c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb',
    'c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb',
    'c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb',
    'c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb',
    'ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb',
    'ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb',
    'ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb',
    'ipcsn_r152_32x2x1-180e_kinetics400-rgb',
    'ircsn_r152_32x2x1-180e_kinetics400-rgb',
    'ipcsn_ig65m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb',
    'ipcsn_sports1m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb',
    'ircsn_sports1m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb',
    'mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb_infer',
    'mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb',
    'mvit-base-p244_32x3x1_kinetics400-rgb', 'mvit-large-p244_40x3x1_kinetics400-rgb',
    'mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb_infer',
    'mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb',
    'mvit-base-p244_u32_sthv2-rgb', 'mvit-large-p244_u40_sthv2-rgb',
    'mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb',
    'slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb',
    'r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb',
    'r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb',
    'slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb',
    'slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb',
    'slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb',
    'slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb',
    'slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb',
    'slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb',
    'slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb',
    'slowonly_r101_8xb16-8x8x1-196e_kinetics400-rgb',
    'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb',
    'slowonly_imagenet-pretrained-r50_8xb16-8x8x1-steplr-150e_kinetics400-rgb',
    'slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-4x16x1-steplr-150e_kinetics400-rgb',
    'slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-8x8x1-steplr-150e_kinetics400-rgb',
    'slowonly_imagenet-pretrained-r50_16xb16-4x16x1-steplr-150e_kinetics700-rgb',
    'slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb',
    'slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb',
    'swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb',
    'swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb',
    'swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb',
    'swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb',
    'swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb',
    'swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb',
    'tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb',
    'tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb',
    'tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb',
    'timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb',
    'timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb',
    'timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb',
    'tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb',
    'tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb',
    'tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb',
    'tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb',
    'tpn-slowonly_imagenet-pretrained-r50_8xb8-8x8x1-150e_kinetics400-rgb',
    'tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb',
    'trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb',
    'trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb',
    'uniformer-small_imagenet1k-pre_16x4x1_kinetics400-rgb',
    'uniformer-base_imagenet1k-pre_16x4x1_kinetics400-rgb',
    'uniformer-base_imagenet1k-pre_32x4x1_kinetics400-rgb',
    'uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb',
    'uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics400-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics400-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics400-rgb',
    'uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics400-rgb',
    'uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics600-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics600-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics600-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics600-rgb',
    'uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb',
    'uniformerv2-base-p16-res224_clip-pre_8xb32-u8_kinetics700-rgb',
    'uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics700-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics700-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics700-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics700-rgb',
    'uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics700-rgb',
    'uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb',
    'uniformerv2-large-p14-res224_clip-pre_u8_kinetics710-rgb',
    'uniformerv2-large-p14-res336_clip-pre_u8_kinetics710-rgb',
    'uniformerv2-base-p16-res224_clip-kinetics710-kinetics-k400-pre_16xb32-u8_mitv1-rgb',
    'uniformerv2-large-p16-res224_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb',
    'uniformerv2-large-p16-res336_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb',
    'vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400',
    'vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400',
    'vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400',
    'vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400',
    'x3d_s_13x6x1_facebook-kinetics400-rgb',
    'x3d_m_16x5x1_facebook-kinetics400-rgb',
    'tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature',
    'bmn_2xb8-400x100-9e_activitynet-feature',
    'bsn_400x100_1xb16_20e_activitynet_feature (cuhk_mean_100)',
    'clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb',
    '2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d',
    '2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d',
    '2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d',
    '2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d',
    '2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d',
    '2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d',
    '2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d',
    '2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d',
    'slowonly_r50_8xb16-u48-240e_gym-keypoint',
    'slowonly_r50_8xb16-u48-240e_gym-limb', 'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint',
    'slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb',
    'slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint',
    'slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint',
    'stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d',
    'stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d',
    'stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d',
    'stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d',
    'stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d',
    'stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d',
    'stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d',
    'stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d',
    'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d'
]
