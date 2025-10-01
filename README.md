# MV-LLMRec: Multi-View Representation Learning with Large Language Models for Recommendation
Minkyung Song, Soyoung Park, Sungsu Lim*

## Franework
<img width="2667" height="1500" alt="framework_2-1" src="https://github.com/user-attachments/assets/75a6b044-5256-4b62-90e8-2e9eea7629c4" />

## Dependencies
Below is a CUDA 11.x example. (If you use a different CUDA/Torch combo, install the matching torch-scatter/torch-sparse wheels.)

```bash
conda create -y -n mv-llmrec python=3.9
conda activate mv-llmrec

# PyTorch (CUDA 11.6 example)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
--extra-index-url https://download.pytorch.org/whl/cu116

# PyG packages (pick wheels that match your Torch/CUDA)
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

pip install pyyaml tqdm
```

## Dataset Structure and Download

**Amazon-book**/ **Amazon-movie**

You can download intent-based semantic embedding files in the following datasets:
Amazon-book/ Amazon-movie [GoogleDrive](https://drive.google.com/drive/folders/1rd2cppCrpoydvI1yvg5sIK2S68sBcn70?usp=sharing)

```plaintext
- amazon_book (/amazon_movie)
|--- trn_mat.pkl # training set (sparse matrix)
|--- val_mat.pkl # validation set (sparse matrix)
|--- tst_mat.pkl # test set (sparse matrix)
|--- usr_emb_np.pkl # user text embeddings
|--- itm_emb_np.pkl # item text embeddings
|--- user_intent_emb_3.pkl # user intent embeddings
|--- item_intent_emb_3.pkl # item intent embeddings
|--- user_conf_emb.pkl # user conformity embeddings
|--- item_conf_emb.pkl # item conformity embeddings
```
Amazon-Book: Uses the preprocessed split provided by RLMRec.
Amazon-Movie: Uses a reprocessed split prepared for this project.

## Train & Evaluate

- **Backbone**
  ```bash
  python encoder/train_encoder.py --model {model_name} --dataset {dataset} --cuda 0

- **RLMRec**
  ```bash
  python encoder/train_encoder.py --model {model_name}_plus --dataset {dataset} --cuda 0
  ```
  ```bash
  python encoder/train_encoder.py --model {model_name}_gene --dataset {dataset} --cuda 0

- **IRLLRec**
  ```bash
  python encoder/train_encoder.py --model {model_name}_int --dataset movie --cuda 0

- **MV-LLMRec**
  ```bash
  python encoder/train_encoder.py --model {model_name}_mv --dataset {dataset} --cuda 0

Hyperparameters:

The hyperparameters of each model are stored in encoder/config/modelconf.

## Acknowledgement

For fair comparison and reproducibility, we reuse parts of the IRLLRec and RLMRec codebases (training/evaluation routines and related utilities). We also adapt user/item profiling and embedding pipeline components. Source repositories:

> [RLMRec](https://github.com/HKUDS/RLMRec)
> 
> [IRLLRec](https://github.com/wangyu0627/IRLLRec)
>
Many thanks to them for providing the training framework and for the active contribution to the open source community.

