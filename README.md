# 環境
- 使用Virtual Machine
	1. [下載ovf檔](https://drive.google.com/file/d/1RvFcVQGo5MKqnvR0p45kvxT2KUtzJct7/view?usp=sharing)
	2. Import至Vmware Workstation
	3. Open Terminal
  ```
  cd Document
  git clone https://github.com/ice1757/cache_imitation_learning.git
  ```
	4. 開啟 Terminal 執行
	```
	conda activate imcr
	```
- 直接安裝
	1. Ubuntu 20.04.3
	2. Install Anaconda and Create Environment with Python 3.7
	```
	conda create --name imcr python=3.7
	conda activate imcr
	```
	3. Install package
	```
	pip install -r requirement.txt
	pip install -e git+https://github.com/openai/baselines.git@ea25b9e8b234e6ee1bca43083f8f3cf974143998#egg=baselines
	```
# Collecting Traces
- [generate program memory trace](https://github.com/google-research/google-research/tree/master/cache_replacement#collecting-traces)
- generate zipf
  ```
  # Current Working directory is cache_imitaion_learning
  cd /cache_replacement/gen_zipf
  python gen_zipf.py <路徑> <address 種類> <accesses 數量> <trace 數量>
  ```
- 相關 trace 會存在"cache_imitation_learning/cache_replacement/policy_learning/cache/traces"

# Cache Size
- 若需要調整 cache size
  1. 需修改 cache_imitation_learning/cache_replacement/policy_learning/cache/configs/default.json
  2. 修改公式：
      - cache size = associativity = cache_line_size * associativity
      - Ex. cache size = 10 需修改  
      capacity = 640  
      associativity = 10

# Cache Replacement Algorithm
- Simple Cache Replacement Algorithm Test
```
## 測試的Trace.csv放入 cache_imitation_learning/cache_replacement/environment/trace
## Current Working directory is cache_imitation_learning/cache_replacement/environment
## start anaconda environment
conda activate imcr
python main.py <方法> <Trace name>
(method including belady, s4lru, lru, belady_nearest_neighbors, random)
```
- 畫圖可參考 cache_imitation_learning/cache_replacement/environment/draw.ipynb
- 修改Cache Environment Hyperparameter
  - Edit cache_imitation_learning/cache_replacement/environment/spec_llc.json
# Belady Cache Simulation Usage

Example usage with Belady's as the policy, and default SPEC cache configs:

```
# Current working directory is cache_imitation_learning
# start anaconda
conda activate imcr
# Execute
python -m cache_replacement.policy_learning.cache.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_belady_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --cache_configs=cache_replacement/policy_learning/cache/configs/eviction_policy/belady.json \
  --memtrace_file=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

Cache hit rate statistics will be logged to tensorboard files in
`/tmp/sample_belady_llc`.

# Parrot Cache Replacement Policy Learning
Train our model (Parrot) to learn access patterns from a particular trace by passing the
appropriate configurations.

Example usage with our full model with all additions, trained and validated on
the sample trace:

```
# Current working directory is cache_imitation_learning
# start anaconda
conda activate imcr
python -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

The number of learned embeddings can be set with the `--model_bindings` flag
(e.g., set to 5000 above).
In our experiments, we set the number of learned embeddings to be the number of
unique memory addresses in the training split.
Hit rate statistics and accuracies will be logged to tensorboard files in
`/tmp/sample_model_llc`.

# Ablation
We also provide commands to run the various ablations reported in the paper.
Training with the byte embedder model:

```
# Current working directory is cache_imitation_learning
python -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_configs=cache_replacement/policy_learning/cache_model/configs/default.json \
  --model_configs=cache_replacement/policy_learning/cache_model/configs/byte_embedder.json \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

Changing the history length to, e.g., 100:

```
# Current working directory is cache_imitation_learning
python -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --model_bindings="sequence_length=100" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

Ablating the reuse distance auxiliary loss:

```
# Current working directory is cache_imitation_learning
python -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

Ablating the ranking loss:

```
# Current working directory is cache_imitation_learning
python -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"log_likelihood\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

Ablating DAgger:

```
# Current working directory is cache_imitation_learning
python -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --dagger_schedule_bindings="initial=0" \
  --dagger_schedule_bindings="update_freq=1000000000000" \
  --dagger_schedule_bindings="final=0" \
  --dagger_schedule_bindings="num_steps=1" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

# Evaluating the Learned Policy

The commands for training the Parrot policy in the previous section also
periodically save model checkpoints.
Below, we provide commands for evaluating the saved checkpoints on a test set.
In the paper, we choose the checkpoint with the highest validation cache hit
rate, which can be done by inspecting the tensorboard files in the training
directory `/tmp/sample_model_llc`.
The following command evaluates the model checkpoint saved after 20000 steps on
the trace `cache_replacement/policy_learning/cache/traces/sample_trace.csv`:

```
# Current working directory is cache_imitation_learning
python -m cache_replacement.policy_learning.cache.main \
  --experiment_base_dir=/tmp \
  --experiment_name=evaluate_checkpoint \
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" \
  --memtrace_file="cache_replacement/policy_learning/cache/traces/sample_trace.csv" \
  --config_bindings="associativity=16" \
  --config_bindings="capacity=2097152" \
  --config_bindings="eviction_policy.scorer.checkpoint=\"/tmp/sample_model_llc/checkpoints/20000.ckpt\"" \
  --config_bindings="eviction_policy.scorer.config_path=\"/tmp/sample_model_llc/model_config.json\"" \
  --warmup_period=0
```

This logs the final cache hit rate to tensorboard files in the directory
`/tmp/evaluate_checkpoint`.

# Reference
https://github.com/google-research/google-research/tree/master/cache_replacement