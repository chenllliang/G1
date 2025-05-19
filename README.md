# G1

Source code for paper G1: Bootstrapping Perception and Reasoning Abilities of Vision-Language Model via Reinforcement Learning

<p align="center">


<img src="assets/image.png" />
</p>



https://github.com/user-attachments/assets/86038b9e-9f78-44f9-b859-82405c0c776e




https://github.com/user-attachments/assets/19833ca3-ce79-4bf8-ba1c-46b93e1fd1e8



## Setup 
```
conda create -n vlmgym python=3.10
conda activate vlmgym
bash setup.sh
```

## Run Parallel Enviroment in VLM-Gym

We provide the evaluation scripts of 4 games using a random policy under `./vlmgym/test`. The 2048 enviroment is based on [gymnasium-2048](https://github.com/Quentin18/gymnasium-2048).

``` bash
cd ./vlmgym/test
python eval_2048.py
```

This would generate the evaluation log file and an image summarizing the curves under `./vlmgym/test/logs` dir and the videos documenting all runs under `./vlmgym/test/videos` dir. All different runs are conducted in parallel. 

It is easy to evaluate different models by implementing the ```custom_policy``` function in the evaluation script, such as using OpanAI class or vLLM. 



<p align="center">
<i>Example 10 parallel random 2048 run curves</i><br>
<img src="assets/test_2048.png" width="50%"  />
</p>

## Customize Difficulties in VLM-Gym

The game config are in ```./vlmgym/sandbox/games/```, for example, you can alter the diffculties of Shisen-Sho game by changing the shape and color settings in ```./vlmgym/sandbox/games/gamematch.py```


## RL Training using VLM-Gym

We provide the RL scripts utilizing the VLM-Gym under `./training/scripts`. Our training is based on [EasyR1](https://github.com/hiyouga/EasyR1/).

For example, to conduct the RL experiments for Shisen-sho game

```bash
cd training
bash scripts/rl_shisensho.sh

# Verified least GPU requirements is 4x80G GPUs.
```

All rollout histroy would be saved under the EasyR1 directory to watch the learning curve.

After training, you can serve the model using vLLM to conduct evaluations with VLM-Gym.


## Citation

If you find our work helpful, please kindly cite

```
coming soon
```

