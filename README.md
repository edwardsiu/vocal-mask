# Vocal Mask CNN

Separate vocals from mixture. 

## General Flow
1. Downsample waveforms to 22050Hz
2. Slice mixture and vocal waveforms into ~290ms frames, with a frame stride of ~11.6ms (derived from hop size (256) / sample rate (22050))
3. Convert each frame to spectrogram via STFT (shape: 513x25)
4. Apply Mel Perceptual Weighting to the spectrogram
5. Normalize spectrogram to range 0-1
5. Select the middle frame of the vocal spectrogram as the target
6. Convert target to binary mask via comparison with `hparams.mask_threshold`
7. Goal is to predict the middle frame of the vocal spectrogram given the mixture spectrogram (with the context of 12 frames before and after the middle frame)
8. Use Binary Cross Entropy Loss function
9. At inference time, slice up the waveform and process each frame one at a time to generate a mask, then apply the mask to the initial spectrogram (before perceptual weighting and normalization)
10. Use inverse STFT to recover the audio from the spectrogram

## Dataset
The DSD100 dataset is used for this project. With 100 songs, sliced into ~290ms windows with a stride of ~11.6ms, and each song averaging about 4 minutes, this gives about 2 million samples for the dataset.

To generate the dataset, download the DSD100 dataset, then move all files in the Mixtures/Dev and Mixtures/Test directories to the Mixtures/ directory, and do the same for Sources/Dev and Sources/Test. Then, delete the Dev and Test subdirectories. 

```python build_dataset.py <dataset root dir> <output dir>```

Run the build_dataset script to slice up the DSD100 dataset and generate the mixture and vocal spectrograms and evaluation waveforms. The striding of the slices is controlled in hparams.stft_stride, where 1 is the smallest possible stride and generates about 2 million slices, 12 produces around 160k slices.

## Training  
Start training by calling:  
```python train.py <path to spectrogram dataset root> --checkpoint=<path to checkpoint>```

The first argument should be the output directory of `build_dataset`.

### Hyper Parameters  
The following parameters can be tuned in `hparams.py`.
- `batch_size`
- `nepochs` - number of epochs to train for
- `fix_learning_rate` - if not `None`, will used a fixed learning rate
- `mask_threshold` - cutoff for determining vocal presence in mask (0-1)
- `ref_level_db` - reference level used for normalizing spectrograms
- `use_preemphasis` - whether or not to apply preemphasis to the waveform
- `res_dims` - number of filters for each residual unit

Additional useful parameters that don't directly affect training:
- `save_every_epoch` - how many epochs to pass before saving checkpoint
- `eval_every_epoch` - how many epochs to pass before running eval step
- `num_evals` - number of eval spectrograms to generate during eval

## Inference  
Run the inference step by calling:  
```python generate.py <path to checkpoint> <path to mixture wav>```

This will generate a vocal wav file in the `generated` directory. `hparams.mask_at_eval` will affect how the mask is applied, where `True` will convert the mask to a binary mask (harder cutoffs), and `False` will leave the mask in the range (0, 1). You may want to generate longer wav files for inference than the normal 2 seconds used for eval step.
