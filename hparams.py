class hparams:

    #--------------     
    # audio processing parameters
    fmin = 20
    fmax = 11025
    fft_size = 1024
    stft_frames = 25
    stft_stride = 1
    hop_size = 256
    win_length = 1024
    sample_rate = 22050
    preemphasis = 0.9
    mask_type = "IBM" # threshold or IBM
    IBM = {
        "alpha": 1.0,
        "theta": 0.5,
        "theta2": 0.1
    }
    threshold = 0.3
    power = {
        "mix": 1,
        "vox": 3
    }
    per_channel_norm = {
        "mix": True,
        "vox": True
    }
    min_level_db = -100
    ref_level_db = 20
    rescaling = False
    rescaling_max = 0.999
    allow_clipping_in_normalization = True
    eval_length = sample_rate*4  # slice size for evaluation samples
    #----------------
    # model parameters
    model_type='mobilenet'  # convnet or resnet18 or resnet34
    init_conv_kernel = (7, 3)
    init_conv_stride = (2, 1)
    init_pool_kernel = None
    init_pool_stride = 2
    kernel = (3, 3)
    # convert target spectrogram to mask at this activity threshold
    mask_threshold = 0.5
    # if False, use soft mask instead of binary
    mask_at_eval = True
    # threshold for masking at inference time
    eval_mask_threshold = 0.5
    #----------------
    #
    #----------------
    # training parameters
    workers = 6
    batch_size = 64
    test_batch_size = 64
    nepochs = 20
    send_loss_every_step = 2000
    save_every_step = 10000
    eval_every_step = 10000
    num_evals = 4  # number of evals to generate
    validation_size = None
    grad_norm = 10
    #learning rate parameters
    initial_learning_rate=1e-3
    lr_schedule_type = 'cca' # or 'noam' or 'step' or 'one-cycle' or 'cca'
    # for noam learning rate schedule
    noam_warm_up_steps = 2000 * (batch_size // 16)
    # for step learning rate schedule
    step_gamma = 0.5
    lr_step_interval = 2000
    # for cyclic learning rate schedule
    min_lr = 1e-3
    max_lr = 2.0
    fine_tune = 0.8
    # for cosine annealing schedule
    M = 4

    optimizer = 'sgd'
    weight_decay = 0.0
    # adam
    adam_beta1=0.9
    adam_beta2=0.999
    adam_eps=1e-8
    amsgrad=False

    # sgd
    momentum = 0.9
    nesterov = True
    #-----------------
