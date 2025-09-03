import sys
sys.path.append("/home/stnikoli/semestral-project-awareness/vae")
from vae_mixture_of_experts import main

if __name__ == "__main__":
    import yaml
    with open('semestral-project-awareness/vae/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = config['train']
    
    n_experts = [15, 20, 25, 30, 35, 40, 45, 50]

    for p in [0.4]:
        for n in n_experts:
            for trial in range(5, 8):
                config['num_experts'] = n
                config['dataset'] = 'quickdraw'
                config['dataset_percentage'] = p
                config['results_dir_suffix'] = f'_{p}_dataset_percentage_{trial}'
                main(config)