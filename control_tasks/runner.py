import os
import subprocess
import asyncio
import yaml
import copy
import numpy as np

config_dir = 'mdl_configs'
template = yaml.load(open(f'{config_dir}/online_l0.yml'))

# data_fracs = [
#     0.0000625,
#     0.000125,
#     0.00025,
#     0.0005,
#     0.001,
#     0.002,
#     0.004,
#     0.008,
#     0.016,
#     0.032,
#     0.0625,
#     0.125,
#     0.25,
#     0.5,
#     1]

data_fracs = np.logspace(np.log10(100000), np.log10(400000), 10) / 1e6
settings = [
    # {'n': 40000, 'model_layer': 0, 'corrupted': True},
    # {'n': 40000, 'model_layer': 1, 'corrupted': True},
    # {'n': 40000, 'model_layer': 2, 'corrupted': True},

    {'n': 40000, 'model_layer': 0, 'corrupted': False, 'seed': 0, 'data_fracs': data_fracs},
    {'n': 40000, 'model_layer': 1, 'corrupted': False, 'seed': 0, 'data_fracs': data_fracs},
    {'n': 40000, 'model_layer': 2, 'corrupted': False, 'seed': 0, 'data_fracs': data_fracs},
]

for setting in settings:
    name = '100k_online_l{model_layer}_n{n}_corrupted{corrupted}_seed{seed}'.format(**setting)
    config = copy.deepcopy(template)
    config['dataset']['dataset_size'] = setting['n']
    config['model']['model_layer'] = setting['model_layer']
    config['probe']['misc']['corrupted_token_percent'] = 1.0 if setting['corrupted'] else 0.0
    config['reporting']['fixed_directory'] = f'../{name}'
    config['regimen']['inds'] = ','.join([str(f) for f in setting['data_fracs']])
    config_path = f'{config_dir}/{name}.yml'
    yaml.dump(config, open(config_path, 'w'))
    setting['config_path'] = config_path

async def run_job(setting, gpu_id):
    command = [
        "python",
        "control_tasks/run_experiment.py",
        setting['config_path'],
        "--seed",
        str(setting['seed']),
    ]
    # subprocess.run(command, shell=False, env={'CUDA_VISIBLE_DEVICES': str(gpu_id), **os.environ})
    print("Dispatching `{}`".format(command))
    proc = await asyncio.create_subprocess_shell(' '.join(command), env={'CUDA_VISIBLE_DEVICES': str(gpu_id), **os.environ})
    stdout, stderr = await proc.communicate()

async def main():
    tasks = (run_job(setting, i % 4) for i, setting in enumerate(settings))
    await asyncio.gather(*tasks)

asyncio.run(main())
