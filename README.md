<div align="center">
  
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="8">UNAD: Universal Anatomy-initialized Noise Distribution Learning Framework Towards Low-dose CT Denoising</font></b>&nbsp;&nbsp;&nbsp;&nbsp;
  </div>
  <img width="80%" src="https://github.com/Nioolek/Nioolek/assets/40284075/8f365df8-a1f7-4c68-95fb-e00379b16fdc"/>
  <div>&nbsp;</div>
</div>

Official repository for **UNAD**: **Un**iversal **A**natomy-initialized Noise **D**istribution Learning Framework Towards Low-dose CT Denoising (accepted by ICASSP2024).
UNAD have achieved state-of-the-art results even without a powerful feature extractor like ViT.

## 🛠️ Installation

UNAD relies on Pytorch and Python 3.6+. To install the required packages, run:

```shell
conda create -n unad python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate unad
pip install -r requirements.txt
```

## 📖 Dataset Preparation

We use the same data processing methods as REDCNN. Please download the 
[2016 NIH-AAPM-Mayo Clinic Low Dose CTGrand Challenge by Mayo Clinic](https://www.aapm.org/GrandChallenge/LowDoseCT/) dataset.
Next, execute the following command to prepare and convert the dataset.

```shell
python prep.py    # Convert dicom file to npy file
```

## 📦 Testing

To replicate our state-of-the-art UNAD results, you can download the model weight from either,
you can download the model weight from either
[Google Drive](https://drive.google.com/file/d/1e5McIAnfKFmp8wJT_qoF6bqteU6s_jWe/view?usp=sharing) or
[Baidu Drive](https://pan.baidu.com/s/1ol7KrMBepATkiOKrfD905w) (password：unad),
and place it to `./work_dirs/unad/` directory. 
Then, execute the following command to test UNAD.

```shell
python main.py config/unad.yaml --test --test_iters 43000
```

The model metrics results will be printed in the console.
The visualization images of the prediction results will be saved in `./work_dirs/unad/fig`.

## 🕹️ Pre-training and Training

The UNAD model training process comprises two phases: pre-training and actual training.

Note: The results in the research paper were obtained from a single GPU. If employing multiple GPUs, a few parameters may need fine-tuning.

### 📚 Pre-training

For single-GPU pretraining, run:

```shell
python main_pretrain.py config/unad_pretrain.yaml
```

For multi-GPU pretraining, run:

```shell
bash ./dist_train.sh config/unad_pretrain.yaml ${GPU_NUM} 
```

### 🧩 Training

After completion of the pre-training phase, you will need to update the `pretrain_path` variable located
in `config/unad.yaml` to the path of the weights saved during the final epoch of the pre-training phase.
Subsequently, run the command below to train UNAD.

For single-GPU training, run:

```shell
python main.py config/unad.yaml
```

For multi-GPU training, run:

```shell
bash ./dist_train.sh config/unad.yaml ${GPU_NUM} 
```

## 🖊️ Citation

If you find this paper useful in your research, please consider citing our paper after it is published.
