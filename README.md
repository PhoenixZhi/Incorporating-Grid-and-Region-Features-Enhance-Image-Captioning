# Incorporating-Grid-and-Region-Features-Enhance-Image-Captioning
This repository contains the reference code for the paper _[Incorporating Grid and Region Features Enhance Image Captioning]()_ (CVPR 2020).

Please cite with the following BibTeX:

```

```
<p align="center">
  <img src="images/m2.png" alt="Meshed-Memory Transformer" width="320"/>
</p>

## Environment setup
Clone the repository and create the `grid_region` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate grid_region
```
Then download spacy data by executing the following command:
```
python -m spacy download en
```

Note: Python 3.7 is required to run our code. 


## Data preparation
To run the code, annotations for the COCO dataset are needed. Please download the annotations file [annotations.zip](https://pan.baidu.com/s/1i4joLltyirMFEiYAy5G0mA), passward：rgsd and extract it.

Detection features are computed with the code provided by [1]. To reproduce our result, please download the COCO features file (https://pan.baidu.com/s/1C6LwpbyJylkTbYk9srljfA ) (~64.0 GB) passward rgsd, in which detections of each image are stored under the `<image_id>` key. `<image_id>` is the id of each COCO image, without leading zeros (e.g. the `<image_id>` for `COCO_val2014_000000037209.jpg` is `37209`), and each value should be a `(N, 2048)` tensor, where `N` is the number of detections. 


## Evaluation
To reproduce the results reported in our paper, download the pretrained model file [Incorporating_Grid_and_Region.pth](https://pan.baidu.com/s/1FDZFojPnejMxJ-s8t8aknw) passward rgsd and place it in the code folder.

Run `python test.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |

#### Expected output
Under `output_logs/`, you may also find the expected output of the evaluation code.


## Training procedure
Run `python train.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--m` | Number of memory vectors (default: 40) |
| `--head` | Number of heads (default: 8) |
| `--warmup` | Warmup value for learning rate scheduling (default: 10000) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |
| `--logs_folder` | Path folder for tensorboard logs (default: "tensorboard_logs")|

For example, to train our model with the parameters used in our experiments, use
```
python train.py --exp_name m2_transformer --batch_size 50 --m 40 --head 8 --warmup 10000 --features_path /path/to/features --annotation_folder /path/to/annotations
```

<p align="center">
  <img src="images/results.png" alt="Sample Results" width="850"/>
</p>

#### References
[1] Nguyen, V.-Q., Suganuma, M., Okatani, T.: Grit: Faster and better image captioning transformer using dual visual features. In: Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVI, pp. 167–184 (2022). Springer
