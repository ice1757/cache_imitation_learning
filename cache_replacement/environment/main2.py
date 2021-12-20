# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Example script for running replacement policies."""
import argparse
import belady
import config as cfg
import environment
import numpy as np
import policy
import s4lru
import tqdm
from absl import app
from absl import logging


def wrt_txt(path, value):
    f = open(path, 'a')
    for i in value:
        f.writelines(str(round(i, 2))+"\n")
        # f.write(str(i))


def main(_):
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "policy_type", help="type of replacement policy to use")
    # args = parser.parse_args()

    trace_path = "./trace/test_30000_16384.csv"
    # trace_path = "./sample_trace.csv"
    config = cfg.Config.from_files_and_bindings(["example_cache_config.json"], [])
    env = environment.CacheReplacementEnv(config, trace_path, 0)

    # if args.policy_type == "belady":
    #     replacement_policy = belady.BeladyPolicy(env)
    # elif args.policy_type == "lru":
    #     replacement_policy = policy.LRU()
    # elif args.policy_type == "s4lru":
    #     replacement_policy = s4lru.S4LRU(config.get("associativity"))
    # elif args.policy_type == "belady_nearest_neighbors":
    #     train_env = environment.CacheReplacementEnv(config, trace_path, 0)
    #     replacement_policy = belady.BeladyNearestNeighborsPolicy(train_env)
    # elif args.policy_type == "random":
    #     replacement_policy = policy.RandomPolicy(np.random.RandomState(0))
    # else:
    #     raise ValueError(f"Unsupported policy type: {args.policy_type}")

    train_env = environment.CacheReplacementEnv(config, trace_path, 0)
    method = dict()
    method["belady"] = (belady.BeladyPolicy(env))
    method["lru"] = (policy.LRU())
    method["s4lru"] = (s4lru.S4LRU(config.get("associativity")))
    method["belady_nearest_neighbors"] = (
        belady.BeladyNearestNeighborsPolicy(train_env))
    method["random"] = (policy.RandomPolicy(np.random.RandomState(0)))
    hit_all = []
    for mm in method:
        state = env.reset()
        total_reward = 0
        steps = 0
        with tqdm.tqdm() as pbar:
            while True:
                action = method[mm].action(state)
                state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                pbar.update(1)
                if done:
                    break
        hit_all.append(total_reward/steps)
        print("Method: {}, Cache hit rate: {:.4f}".format(
            mm, (total_reward / steps)))

    wrt_txt('3.txt', hit_all)

if __name__ == "__main__":
    app.run(main)