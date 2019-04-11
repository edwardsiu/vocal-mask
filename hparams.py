class hparams:

    #--------------     
    # audio processing parameters
    num_mels = 256
    fmin = 125
    fmax = 11025
    fft_size = 1024
    stft_frames = 25
    stft_stride = 6
    hop_size = 256
    win_length = 1024
    sample_rate = 22050
    use_preemphasis = True # apply preemphasis transformation to waveform
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20
    lws_mode = 'speech' # alternatively 'music'
    rescaling = False
    rescaling_max = 0.999
    allow_clipping_in_normalization = True
    trim = False # whether to cut silence from the ends of the waveform
    trim_thresh = 80 # how much below max db to trim waveform
    mixture_fname = "mixture.wav"
    vocal_fname = "vocals.wav"
    #----------------
    #
    #----------------
    # model parameters
    rnn_dims = 800
    fc_dims = 512
    pad = 2
    # note upsample factors must multiply out to be equal to hop_size, so adjust
    # if necessary (i.e 4 x 4 x 16 = 256)
    upsample_factors = (4, 4, 16)
    compute_dims = 128
    res_out_dims = 128
    res_blocks = 10
    y_tsfm = None
    #----------------
    #
    #----------------
    # training parameters
    batch_size = 64
    nepochs = 106
    save_every_epoch = 5
    eval_every_epoch = 5
    train_test_split = 0.1 # reserve 10% of data for validation
    # seq_len_factor can be adjusted to increase training sequence length (will increase GPU usage)
    seq_len_factor = 5
    seq_len = seq_len_factor * hop_size
    grad_norm = 10
    #learning rate parameters
    initial_learning_rate=5e-4
    lr_schedule_type = 'step' # or 'noam'
    # for noam learning rate schedule
    noam_warm_up_steps = 2000 * (batch_size // 16)
    # for step learning rate schedule
    step_gamma = 0.1
    lr_step_interval = 5000

    adam_beta1=0.9
    adam_beta2=0.999
    adam_eps=1e-8
    amsgrad=False
    weight_decay = 0.0
    #fix_learning_rate = 5e-6 # modify if one wants to use a fixed learning rate, else set to None to use noam learning rate
    fix_learning_rate = 1e-4
    #-----------------
