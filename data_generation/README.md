# Label Preserving / Breaking generation with Stable Diffusion

## Setup

1. Clone [Stable Diffusion](https://github.com/CompVis/stable-diffusion) locally and setup the environment

If running on virtual environment (conda)
```
conda install python=3.8.5 pip=20.3 cudatoolkit=11.3 pytorch=1.11.0 torchvision=0.12.0 numpy=1.19.2 -c pytorch -c defaults
pip install -r requirements.txt
```

If running on your own machine follow instructions on [Stable Diffusion repo.](https://github.com/CompVis/stable-diffusion) to
start a new environment from scratch

<br>

2. Make sure to have the [dataset](https://uvaauas.figshare.com/articles/dataset/Low-Resource_Image_Transfer_Evaluation_Benchmark/25577145?file=45571590) you want to apply augmentation on.

<br>

3. Move img2img_circuit.py inside stable diffusion folder

<br>

4. Download SD checkpoint :
```
wget --content-disposition "https://huggingface.co/CompVis/stable-diffusion-v-1-1/resolve/main/sd-v1-1.ckpt"  
```

<br>

5. [**Adjust absolute paths**](https://github.com/OmarFacchini/LoRa-challengingDatasets/blob/fd4c41b7647d9d14f9a3284d6cfb993b58315cce/data_generation/img2img_circuit.py#L3#L9) at the top of "img2img_circuit.py" with your current setup

<br>

## Generate samples

Label preserving
```
python3 img2img_circuit.py --n_samples 3
```

Label breaking
```
python3 img2img_circuit.py --label_breaking --n_samples 3
```
