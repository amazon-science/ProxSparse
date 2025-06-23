# ProxSparse: Regularized Learning of Semi-Structured Sparsity Masks for Pretrained LLMs

<p align="center">
  <img src="assets/semi-structured-sparsity-pattern.png" alt="2-4 Structured Sparsity" width="750">
  <br>
  <em>Semi-Structured sparsity (eg.: 2-4 sparsity) provides a middle ground between strctured puning (removing entire sub-structures like neurons or attention heads) and non-uniform unstructured sparsity. However, finding the optimal 2:4 mask is NP-hard and non-differentiable. <a href="https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/">[Image Source]</a> </em>
</p>

ProxSparse is a learning-based framework for semi-structured (2:4) pruning of Large Language Models using only a few hundred calibration samples. The optimal mask selection is enabled by regularized optimization, which transforms the rigid, non-differentiable mask selection process into a smoother optimization procedure, allowing gradual mask exploration with flexibility. ProxSparse does not involve additional weight updates once the mask is determined.

🔗  You can find our paper (ICML'25) [here](https://arxiv.org/abs/2502.00258).

## 🛠 Setup Instructions

The required environment to run ProxSparse is stored in ``requirement.txt``. You can run the below command to install them.

``` 
conda create --name proxsparse python==3.10
conda activate proxsparse
pip install -r requirement.txt
```

## 💾 Model checkpoints produced by ProxSparse

We release our 2:4 pruned models induced by ProxSparse in the Huggingface repository [aladinggit/proxsparse_models](https://huggingface.co/aladinggit/proxsparse_models/tree/main). The repo contains 2:4 pruned checkpoints of Llama-2-7b, Llama-2-13b, Llama-3.1-8b, Mistral-v0.1-7b, Mistral-v0.3-7b, Openllama-v2-7b and Qwen-2.5-14b. 

The downloading scripts will download those pruned models from the huggingface repository.

```python proxsparse_pruned_model_download.py```

``eval/ppl.py`` contains a standard evaluation for C4/wikitext perplexity. 

To run a quick evaluation on Wikitext perplexity on those checkpoints: ```bash script/eval_pruned_prox_model.sh```

## 🧮 Learning semi-structured 2:4 mask with ProxSparse

Here we provide steps to prune models with ProxSparse. We use ``transformers``, ``accelerate`` and ``trl`` package as the main training framework.
The core ProxSparse operator can be found in `end-to-end/prox_op.py`. If you want to integrate ProxSparse operator into a different training framework, we provide more information regarding how to do so in `end-to-end/doc.txt`.

The ``script``directory contains scripts to run ProxSparse and learn the mask. For example, `bash learn_mask_llama2_7b_prox.sh` will learn mask for llama-2-7b model. To learn with Qwen-2.5-14b model, run `bash learn_mask_qwen_2.5_14b_prox.sh`

A description of the parameters can be found in the entry point ``end-to-end/main.py``. A brief description of the arguments is also provided below:

`*_prox.sh` will launch training with only the $\lambda_{1}$ (semi-structured regularizer)

- ``--model_dir and model_subdir``: the name of the model.
- ``--lambda``: the hyperparameter of the $\lambda_{1}$ denoting the strength of the semi-structured regularizer.
- ``--batch_size``: data batch size.
- ``--ctx_len``: the context length of the data used in training process.
- ``--samples``: number of data used in training process.
- ``--lr``: learning rate.

`*_full.sh` will launch training with both the $\lambda_{1}$ (semi-structured regularizer) and $\lambda_{2}$ (frozen weight regularizer).

- ``--lambda2_``: the hyperparameter of the $\lambda_{2}$ denoting the strength of the frozen weight regularizer.
- ``--project_lambda2``: set this to 0 (by default) for training only with $\lambda_{1}$ (semi-structured regularizer). Set this to 1 raining with both the $\lambda_{1}$ (semi-structured regularizer) and $\lambda_{2}$ (frozen weight regularizer).
- ``--epsilon``: the epsilon term in $\lambda_{2}$ to avoid numerical instability.

Each script contains three main functions: learning with ProxSparse, extracting the mask, applying, and evaluating the mask. More information about the script can be found in ``script/doc.txt``. The optimal configuration for the different models has been set in the script (refer our paper). To extract masks for additional models that are not covered by these scripts, modify the model name and other corresponding configurations to run them.


## 📊 Baselines

The baseline directory is the implementation from [Wanda](https://github.com/locuslab/wanda) repository. For details regarding additional baselines ([AdmmPrune](https://github.com/fmfi-compbio/admm-pruning), [OWL](https://github.com/luuyin/OWL) and [AlphaPrune](https://github.com/haiquanlu/AlphaPruning)), please refer to their original implementation!

## 🧑‍💻 Contributing

This project welcomes contributions and suggestions, see [CONTRIBUTING.md](./CONTRIBUTING.md) for details. This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact opensource-codeofconduct@amazon.com with any additional questions or comments.

## 📄 License

This project is licensed under the [`Apache 2.0 license`](https://opensource.org/licenses/Apache-2.0). 

## 📚 Citation

If you might find our work useful, please cite:
```
@article{liu2025proxsparse,
  title={ProxSparse: Regularized Learning of Semi-Structured Sparsity Masks for Pretrained LLMs},
  author={Liu, Hongyi and Saha, Rajarshi and Jia, Zhen and Park, Youngsuk and Huang, Jiaji and Sabach, Shoham and Wang, Yu-Xiang and Karypis, George},
  journal={arXiv preprint arXiv:2502.00258},
  year={2025}
}

```
