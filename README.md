# CaFA
Code for ForeCasting with Factorized Attention (CaFA) [arxiv](https://arxiv.org/abs/2405.07395)

<div style style=”line-height: 20%” align="center">
<img src="https://github.com/BaratiLab/CaFA/blob/main/assets/u10m.gif" width="600">
</div>

### 1. Environment

We provided a yaml file for the conda environment used for this project. </br>

```bash
conda env create -f environment.yml
```

### 2. Data

The ERA5 reanalysis data courtesy is under [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/#!/home). [WeatherBench2](https://weatherbench2.readthedocs.io/en/latest/index.html) provides processed version of it with different resolutions (in ```.zarr``` format). </br>
Data and normalization statistics in ```.npy``` format that is used in this project are provided below, which is derived from WeatherBench2's dataset.

| Resolution       |Train range | Validation range | link   |
|---------------|---------|------------|------------------------------------------------------|
|  240 x 121  |1979-2018 |2019| [link](https://huggingface.co/datasets/JleeOfficial/ERA5-240x121-1979-2018/tree/main) |
| 64 x 32 |1979-2015 |2016| [link](https://huggingface.co/datasets/JleeOfficial/ERA5-64x32-1979-2015) |

(Note that the final model on 240 x 121 is trained with year 2019 in training data.)

### 3. Training

The configuration for training are provided under ```configs``` directory. 
Training a 100M parameter CaFA on 64 x 32 resolution will take around 15 hr for stage 1 and 50 hr for stage 2 on a 3090.

Stage 1 training example (on 64 x 32 resolution):
```bash
bash run_stage1.sh
```

Stage 2 fine tuning example (on 64 x 32 resolution):
```bash
bash run_stage2.sh
```
The pre-trained checkpoints can be downloaded through belowing links

| Resolution       |Train range | # of Parameter | link   |
|---------------|---------|------------|------------------------------------------------------|
|  240 x 121  |1979-2019 |~200M| [link](https://huggingface.co/datasets/JleeOfficial/Trained-model-ckpt/blob/main/ema_last240121.pth) |
| 64 x 32 |1979-2015 |~100M| [link](https://huggingface.co/datasets/JleeOfficial/Trained-model-ckpt/blob/main/ema_last6432.pth) |

### 4. Inference

To run model inference on processed npy files, please refer to ```validate_loop()``` function under ```dynamics_training_loop.py```.

Here we povide a demo ipynb to showcase how to run the model on weatherbench2's data, check: ```inference_demo.ipynb```.

### Acknowledgement

The ERA data used in the project is from European Centre for Medium-Range Weather Forecasts. WeatherBench2 has provided processed and aggregated versions, which is publicly available at [link](https://console.cloud.google.com/storage/browser/weatherbench2?pli=1).

The spherical harmonics implementation is taken from [se3_transformer_pytorch/spherical_harmonics.py](https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/spherical_harmonics.py).

### Reference

If you find this project useful, please kindly consider citing our work:
```
@misc{li2024cafa,
      title={CaFA: Global Weather Forecasting with Factorized Attention on Sphere}, 
      author={Zijie Li and Anthony Zhou and Saurabh Patil and Amir Barati Farimani},
      year={2024},
      eprint={2405.07395},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
