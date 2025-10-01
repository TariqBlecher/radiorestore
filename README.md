
# mldeconv

## Dataset
```
python drivers/create_sources.py --output_dir source/eso_5k
```

## Training
```
python drivers/deconvolution.py --output_dir results/test
```

## Inference
```
python drivers/predict_windowed.py --output_dir results/test
```

## Hardware

Trained on Tesla V100 32GB

```
FP16 (half)
28.26 TFLOPS (2:1)
FP32 (float)
14.13 TFLOPS
FP64 (double)
7.066 TFLOPS (1:2)
```

## File Structure

input

```
sources/
├── eso
│   ├── eso137_psf_centered.fits
│   └── test1.3to1.5GHz_I_main_dirty_time0000_mfs.fits
├── eso_2k_fp32
│   ├── dirty_sky.npy
│   └── true_sky.npy
└── eso_2k_fp64
    ├── dirty_sky.npy
    └── true_sky.npy
```

output
```
results/unet_eso_2k_fp32/
├── args.json
├── deconv_model.pt
├── iter
├── metrics.pt
├── py_files/
└── training.log
```

args.json
```
{
  "output_dir": "results/unet_eso_2k_fp32",
  "true_sky_file": "sources/eso_2k_fp32/true_sky.npy",
  "dirty_sky_file": "sources/eso_2k_fp32/dirty_sky.npy",
  "psf_file": "sources/eso/eso137_psf_centered.fits",
  "learning_rate": 0.0001,
  "epochs": 100,
  "model": "unet",
  "batch_size": 8,
  "val_split": 0.2,
  "use_bn": 1,
  "deep_supervision": 1,
  "dtype": "float32",
  "device": "cuda"
}
```