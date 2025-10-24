<div align="center">

# Scheduling Your LLM Reinforcement Learning with Reasoning Trees

</div>




### Introduction

Using Reinforcement Learning with Verifiable Rewards (RLVR) to optimize Large Language Models (LLMs) can be conceptualized as progressively editing a query's `Reasoning Tree'. This process involves exploring nodes (tokens) and dynamically modifying the model's policy at each node. When combined with data scheduling, this process yields further gains in data efficiency and accuracy.
However, existing RLVR data scheduling methods typically rely on path-based metrics to rank queries, overlooking the reasoning tree structures of these queries.
In this paper, we introduce a novel metric, namely Reasoning Score r-score, which measures the query's learning difficulty based on the structure of its reasoning tree.
Based on the r-score, we propose the Reasoning Tree Schedule Re-Schedule, a scheduling algorithm that constructs a curriculum progressing from structurally simple (high r-score) to complex (low r-score) queries.


- **We introduce the Reasoning Score (r-score)** , a new tree-based metric that measures a query's learning efficiency rather than its path-based solution accuracy.
    
- **We propose Re-Schedule**, a data scheduling algorithm that uses the r-score to create an effective, easy-to-hard curriculum for RLVR.


Experiments on six math-reasoning benchmarks show that {Re-Schedule} significantly improves average accuracy, achieving gains of up to 3.2\%.
These strong results validate our approach and demonstrate that a structural understanding of the reasoning tree provides a more powerful and principled foundation for RLVR data scheduling.






### Getting Started


We use exactly the same environment configurations as the official verl codebase.

* **Install:** [https://verl.readthedocs.io/en/latest/start/install.html](https://verl.readthedocs.io/en/latest/start/install.html)
* **Quick Start:** [https://verl.readthedocs.io/en/latest/start/quickstart.html](https://verl.readthedocs.io/en/latest/start/quickstart.html)

Environment setup
```bash
pip install git+ssh://git@github.com/volcengine/verl.git@01ef7184821d0d7844796ec0ced17665c1f50673
```


### Datasets
We use public dataset [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) for training, and six public math benchmarks for validation. 
All datasets are provided in folder `Re-Schedule/datasets`.


### Base Model
We use [Qwen](https://huggingface.co/Qwen/collections) series model for training.
One can download the models from huggingface, for example,
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-Math-7B', cache_dir='Qwen2.5-Math-7B')"
```


### Training
The training scripts are fully inherited from the standard GRPO training.

We provide ready-to-run scripts:
```
cd Re-Schedule/reasoning_tree
bash run_batch_vllm.sh
```
The hyperparameters involved in reasoning tree construction are listed in the script.
These hyperparameters can be adjusted to construct reasoning trees of different scales and structures.
`run_batch_vllm.sh` supports parallel tree construction by setting GPU_DEVICES, START_INDEX, and END_INDEX, and running the script on multiple devices separately, which accelerates the tree-building process.
The reasoning tree will be saved in `Re-Schedule/reasoning_tree/eval_logs` 


After constructing the reasoning tree, the R-score can be computed, and the selected dataset can be ranked by R-score in descending order.
```
cd Re-Schedule/reasoning_tree
bash run_add_metrics.sh
```
In addition to calculating the R-score from the perspective of fixing nodes (where `METRIC_TYPE="parent_selection_improvement_topM_sum"`), it can also be computed from the pruning perspective by modifying `METRIC_TYPE="pruning_improvement_topM_sum"`.




Next, by modifying the `train_path` in the script to point to the generated dataset with the ranking (metric), the training script can be run:
```
cd Re-Schedule/run
bash Re_Schedule_sigmoid.sh
bash Re_Schedule_linear.sh
```
For direct training, we provide an example of a reasoning tree with the default settings `Re-Schedule/reasoning_tree/eval_logs/depth_4_branch_4`, along with the corresponding dataset ranked by metrics R-score `Re-Schedule/datasets/data_ranked.parquet`, where the attribute 'metric' represents the sample's ranking percentage.


The core implementation involves converting the metric rankings into data weights, as provided in the file: `Re-Schedule/verl/verl/trainer/ppo/ray_trainer.py`.


Note that we run all experiments using 8 H20s.
If one want to launch distributed tasks, please refer to the instruction of [verl](https://github.com/volcengine/verl/tree/gm-tyx/puffin/main).


### Evaluation
We provide the evaluation codebase integrated in the verl infra.
Please refer to script eval.sh for evaluation scripts on our [released model](https://huggingface.co/zzzzzzzzzzhao/Re-Schedule).
```
cd Re-Schedule/run
bash eval.sh
```



## Acknowledgement

We build on [verl](https://github.com/volcengine/verl) and qwen math-reasoning evaluation protocols.
All competitors in can be easily implemented or are already implemented in [verl](https://github.com/volcengine/verl).

---





## Citation

```bibtex
@article{hao2025rethinking,
  title={Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective},
  author={Hao, Zhezheng and Wang, Hong and Liu, Haoyang and Luo, Jian and Yu, Jiarui and Dong, Hande and Lin, Qiang and Wang, Can and Chen, Jiawei},
  journal={arXiv preprint arXiv:2510.10150},
  year={2025}
}
```



## Contact

* Zhezheng Hao — [haozhezheng@zju.edu.cn](mailto:haozhezheng@zju.edu.cn)

