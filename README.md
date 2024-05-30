# CaFA
Code for ForeCasting with Factorized Attention (CaFA)

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

(The final model on 240 x 121 is trained with year 2019 in training data.)

### 3. Training

The configuration for training are provided under ```configs``` directory. 

Stage 1 training example (on 64 x 32 resolution):
```bash
bash run_stage1.sh
```

Stage 2 fine tuning example (on 64 x 32 resolution):
```bash
bash run_stage2.sh
```


### 4. Inference

