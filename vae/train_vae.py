from vae_mixture_of_experts import main

if __name__ == "__main__":
    ### Experiment with varying dataset percentage (Fig. 6 in the paper)
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = config['train']
    
    n_experts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for p in [0.05, 0.1, 0.2, 0.4, 1.0]:
        for n in n_experts:
            for trial in range(0, 5):
                config['num_experts'] = n
                config['dataset'] = 'quickdraw'
                config['dataset_percentage'] = p
                config['results_dir_suffix'] = f'_{p}_dataset_percentage_{trial}'
                main(config)