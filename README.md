# Feature EndoGaussian: Feature Distilled Gaussian Splatting in Surgical Deformable Scene Reconstruction

## arXiv Preprint

### [Project Page]()| [arXiv Paper]()


[Kai Li](https://www.linkedin.com/in/kaii-li/)<sup>1*</sup>, [Junhao Wang](https://www.linkedin.com/in/junhao-wang-4ba488200/)<sup>1*</sup>,
[William Han](https://willxxy.github.io)<sup>2*</sup>, [Ding Zhao](https://scholar.google.com/citations?user=z7tPc9IAAAAJ&hl=en)<sup>2*</sup>

<sup>1</sup>Department of Engineering Science, University of Toronto; <sup>2</sup>Computer Science Department, Carnegie Mellon University;

<sup>\*</sup> Equal Contributions. <sup>✉</sup> Corresponding Author. 

-------------------------------------------


## Commands to run the code

### Get correct CUDA version (If you get error)

```
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```


### Environment Setup

```
git submodule update --init --recursive

conda create -n EndoGaussian python=3.10

conda activate EndoGaussian

pip install -r requirements.txt

pip install -e submodules/depth-diff-gaussian-rasterization

pip install -e submodules/simple-knn
```

### Get the embeddings

```
cd encoders/sam_encoder
pip install -e .
```
Download the following same encoders

ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Place it in `encoders/sam_encoder/checkpoints`

```
cd encoders/sam_encoder
python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input <dataset_path>/images  --output <data_path>/sam_embeddings
```


### For training EndoNerf Pulling:

Use below to try training directly on endonerf data—pulling soft tissues dataset

```
python train.py -s ../pulling_soft_tissues --port 6017 --expname endonerf/pulling --configs arguments/endonerf/pulling.py
```


### Rendering

```
python render.py --model_path output/endonerf/pulling  --skip_train --skip_video --configs arguments/endonerf/pulling.py
```


### Evaluation

```
python metrics.py --model_path output/endonerf/pulling
```

### For training EndoNerf Cutting:

Use below to try training directly on endonerf data—cutting dataset

```
python train.py -s ../cutting_tissues_twice --port 6017 --expname endonerf/cutting --configs arguments/endonerf/cutting1.py
```


### Rendering

```
python render.py --model_path output/endonerf/cutting  --skip_train --skip_video --configs arguments/endonerf/cutting1.py
```


### Evaluation

```
python metrics.py --model_path output/endonerf/cutting1.py
```



expected folder structure 

```
train 
 --images
 --sam_embeddings
 --sparse/0
   --cameras.bin
   --images.bin
   --points3D.bin
   --points3D.ply
   --project.ini
```
