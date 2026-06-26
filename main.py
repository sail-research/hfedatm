import time
import datetime
import gc
import argparse
import torch
import torch.cuda
import src.server as server_methods
import src.client as client_methods
from src.server import *
from src.client import *
import src.datasets as my_datasets
# from dataclasses import dataclass
from src.splitter import *
from src.hierarchy import assign_clients_to_stations
from src.partitioning import build_hierarchical_partition
from src.utils import *
from src.dataset_bundle import *
from wilds.common.data_loaders import get_eval_loader
from wilds import get_dataset

import wandb
from wandb_env import WANDB_ENTITY, WANDB_PROJECT
"""
The main file function:
1. Load the hyperparameter dict.
2. Initialize logger
3. Initialize data (preprocess, data splits, etc.)
4. Initialize clients. 
5. Initialize Server.
6. Register clients at the server.
7. Start the server.
"""


def resolve_server_class(method_name):
    try:
        return getattr(server_methods, method_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown server_method: {method_name}") from exc


def resolve_client_class(method_name):
    client_aliases = {
        "FedAvg": "ERM",
    }
    resolved_name = client_aliases.get(method_name, method_name)
    try:
        return getattr(client_methods, resolved_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown client_method: {method_name} (resolved to {resolved_name})") from exc


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def main(args):
    hparam = vars(args)
    config_file = args.config_file
    with open(config_file) as fh:
        config = json.load(fh)
    hparam.update(config)
    wandb_project = WANDB_PROJECT
    # setup WanDB
    if not args.no_wandb:
        hparam['wandb'] = True
        wandb.init(project=wandb_project,
                    entity=WANDB_ENTITY,
                    name=hparam.get("id"),
                    config=hparam)
        wandb.define_metric("comm_round")
        wandb.define_metric("eval/*", step_metric="comm_round")
        wandb.define_metric("loss/*", step_metric="comm_round")
        wandb.run.log_code()
    else:
        hparam['wandb'] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    seed = hparam['seed']
    set_seed(seed)
    data_path = hparam['data_path']
    os.makedirs(data_path, exist_ok=True)
    if hparam.get("save_checkpoints", False):
        if not os.path.exists(data_path + "opt_dict/"): os.makedirs(data_path + "opt_dict/")
        if not os.path.exists(data_path + "sch_dict/"): os.makedirs(data_path + "sch_dict/")
        if not os.path.exists(data_path + "models/"): os.makedirs(data_path + "models/")

    # optimizer preprocess
    if hparam['optimizer'] == 'torch.optim.SGD':
        hparam['optimizer_config'] = {'lr':hparam['lr'], 'momentum': hparam['momentum'], 'weight_decay': hparam['weight_decay']}
    elif hparam['optimizer'] == 'torch.optim.Adam' or hparam['optimizer'] == 'torch.optim.AdamW':
        hparam['optimizer_config'] = {'lr':hparam['lr'], 'eps': hparam['eps'], 'weight_decay': hparam['weight_decay']}

    # initialize data
    if hparam['dataset'].lower() == 'pacs':
        if hparam.get('pacs_tensor_cache'):
            dataset = my_datasets.CachedPACS(
                version='1.0',
                root_dir=hparam['dataset_path'],
                download=True,
                split_scheme=hparam["split_scheme"],
                cache_dir=hparam.get('pacs_tensor_cache_dir'),
            )
        else:
            dataset = my_datasets.PACS(
                version='1.0',
                root_dir=hparam['dataset_path'],
                download=True,
                split_scheme=hparam["split_scheme"],
            )
    elif hparam['dataset'].lower() == 'officehome':
        dataset = my_datasets.OfficeHome(version='1.0', root_dir=hparam['dataset_path'], download=True, split_scheme=hparam["split_scheme"])
    elif hparam['dataset'].lower() == 'femnist':
        dataset = my_datasets.FEMNIST(version='1.0', root_dir=hparam['dataset_path'], download=True)
    elif hparam['dataset'].lower() == 'celeba':
        dataset = get_dataset(dataset="celebA", root_dir=hparam['dataset_path'], download=True)
    else:
        dataset = get_dataset(dataset=hparam["dataset"].lower(), root_dir=hparam['dataset_path'], download=True)
    # Dataset bundles inspect these optional attributes to select a lightweight
    # backbone/input size while preserving the historical ResNet-224 default.
    setattr(dataset, "_hfed_backbone", hparam.get("model_arch") or hparam.get("backbone", "resnet50"))
    setattr(dataset, "_hfed_image_size", hparam.get("image_size", hparam.get("input_size", 224)))
    # if server_config['algorithm'] == "FedDG":
    #     # make it easier to hash fourier transformation
    #     indices = torch.arange(len(dataset)).reshape(-1,1)
    #     new_metadata_array = torch.cat((dataset.metadata_array, indices), dim=1)
    #     dataset._metadata_array = new_metadata_array
    if hparam['client_method'] == "FedSR":
        ds_bundle = eval(hparam["dataset"])(dataset, probabilistic=True)
    else:
        if hparam['dataset'].lower() == 'py150' or hparam['dataset'].lower() == 'civilcomments':
            ds_bundle = eval(hparam["dataset"])(dataset, probabilistic=False)
        else:
            ds_bundle = eval(hparam["dataset"])(dataset, probabilistic=False)
    use_cached_pacs = hparam.get('pacs_tensor_cache') and hparam['dataset'].lower() == 'pacs'
    train_transform = None if use_cached_pacs else ds_bundle.train_transform
    test_transform = None if use_cached_pacs else ds_bundle.test_transform
    if hparam['server_method'] == "FedDG":
        if use_cached_pacs:
            raise NotImplementedError("PACS tensor cache is not wired for Fourier FedDG training.")
        if hparam["dataset"].lower() == "iwildcam":
            dataset = my_datasets.FourierIwildCam(root_dir=hparam['dataset_path'], download=True)
            total_subset = dataset.get_subset('train', transform=test_transform)
        elif hparam["dataset"].lower() == "pacs":
            dataset = my_datasets.FourierPACS(root_dir=hparam['dataset_path'], download=True, split_scheme=hparam["split_scheme"])
            total_subset = dataset.get_subset('train', transform=test_transform)
        elif hparam["dataset"].lower() == "celeba":
            dataset = my_datasets.FourierCelebA(root_dir=hparam['dataset_path'], download=True, split_scheme=hparam["split_scheme"])
            total_subset = dataset.get_subset('train', transform=test_transform)
        elif hparam["dataset"].lower() == "camelyon17":
            dataset = my_datasets.FourierCamelyon17(root_dir=hparam['dataset_path'], download=True, split_scheme=hparam["split_scheme"])
            total_subset = dataset.get_subset('train', transform=test_transform)
        elif hparam["dataset"].lower() == "femnist":
            dataset = my_datasets.FourierFEMNIST(root_dir=hparam['dataset_path'], download=True, split_scheme=hparam["split_scheme"])
            total_subset = dataset.get_subset('train', transform=test_transform)
        else:
            raise NotImplementedError
    else:
        total_subset = dataset.get_subset('train', transform=train_transform)

    testloader = {}
    for split in dataset.split_names:
        if split != 'train':
            ds = dataset.get_subset(split, transform=test_transform)
            if len(ds) == 0:
                print(f"Skipping empty eval split: {split}")
                continue
            dl = get_eval_loader(loader='standard', dataset=ds, batch_size=hparam["batch_size"])
            testloader[split] = dl

    
    sampler = RandomSampler(total_subset, replacement=True)
    global_dataloader = DataLoader(total_subset, batch_size=hparam["batch_size"], sampler=sampler)
    # # DS
    # out_test_dataset, test_train = RandomSplitter(ratio=0.5, seed=seed).split(out_test_dataset)
    # out_test_dataset.transform = ds_bundle.test_transform
    # out_test_dataloader = get_eval_loader(loader='standard', dataset=out_test_dataset, batch_size=global_config["batch_size"])
    # if global_config['cheat']:
    #     total_subset = concat_subset(total_subset, test_train)
    # training_datasets = [total_subset]
    # print(len(total_subset), len(in_validation_dataset), len(lodo_validation_dataset), len(in_test_dataset), len(out_test_dataset))
    partition_result = build_hierarchical_partition(
        dataset.get_subset("train"),
        domain_field=ds_bundle.groupby_fields,
        transform=train_transform,
        hparam=hparam,
    )
    training_datasets = partition_result.client_datasets
    station_client_indices = partition_result.station_client_indices

    # initialize client
    clients = []
    client_cls = resolve_client_class(hparam["client_method"])
    for k in tqdm(range(hparam["num_clients"]), leave=False, disable=hparam.get("disable_tqdm", False)):
        client = client_cls(k, device, training_datasets[k], ds_bundle, hparam)
        clients.append(client)
    message = f"successfully initialize all clients!"
    logging.info(message)
    del message; gc.collect() 

    # initialize server (model should be initialized in the server. )
    server_cls = resolve_server_class(hparam["server_method"])
    central_server = server_cls(device, ds_bundle, hparam)
    if hparam['server_method'] == "FedDG":
        central_server.set_amploader(global_dataloader)
    if hparam['start_epoch'] == 0:
        central_server.setup_model(None, 0)
    else:
        central_server.setup_model(hparam['resume_file'], hparam['start_epoch'])
    central_server.register_clients(clients)
    if hasattr(central_server, "register_stations"):
        if not station_client_indices:
            station_client_indices = assign_clients_to_stations(
                training_datasets,
                num_stations=hparam["num_stations"],
                assignment=hparam["station_assignment"],
                domain_field=ds_bundle.groupby_fields,
                seed=seed,
            )
        central_server.register_stations(station_client_indices)
    central_server.register_testloader(testloader)
    # do federated learning
    central_server.fit()
    
    # bye!
    message = "...done all learning process!\n...exit program!"
    logging.info(message)
    time.sleep(3)
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedDG Benchmark')
    parser.add_argument('--config_file', help='config file', default="config.json")
    parser.add_argument('--no_wandb', default=False, action="store_true")
    parser.add_argument('--seed', default=1001, type=int)
    parser.add_argument('--num_clients', default=1, type=int)
    parser.add_argument('--num_stations', default=1, type=int)
    parser.add_argument('--station_rounds', default=1, type=int)
    parser.add_argument('--station_assignment', default='contiguous')
    parser.add_argument('--station_fraction', default=1, type=float)
    parser.add_argument('--station_client_fraction', default=1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--iid', default=1, type=float)
    parser.add_argument('--partition_method', default='paper_lambda', choices=[
        'paper_lambda', 'paper_lambda_clustered', 'paper_lambda_mixed',
        'hds_eta_lambda', 'hds_dirichlet', 'hds_inter', 'hds_intra', 'hds_severe',
        'hds_quantity', 'hds_partial', 'hds_label', 'hds_full',
    ])
    parser.add_argument('--partition_eta', default=None, type=float)
    parser.add_argument('--partition_lambda', default=None, type=float)
    parser.add_argument('--partition_alpha_station', default=1.0, type=float)
    parser.add_argument('--partition_alpha_client', default=1.0, type=float)
    parser.add_argument('--partition_quantity_mode', default='none', choices=['none', 'lognormal', 'dirichlet'])
    parser.add_argument('--partition_station_size_sigma', default=0.0, type=float)
    parser.add_argument('--partition_client_size_sigma', default=0.0, type=float)
    parser.add_argument('--partition_station_size_alpha', default=10.0, type=float)
    parser.add_argument('--partition_client_size_alpha', default=10.0, type=float)
    parser.add_argument('--partition_min_client_samples', default=1, type=int)
    parser.add_argument('--partition_resample_until_nonempty', default=True, type=str2bool)
    parser.add_argument('--partition_max_resample_attempts', default=100, type=int)
    parser.add_argument('--partition_report_dir', default=None)
    parser.add_argument('--partition_save_report', default=True, type=str2bool)
    parser.add_argument('--partition_plot', default=False, type=str2bool)
    parser.add_argument('--server_method', default='FedAvg')
    parser.add_argument('--fraction', default=1, type=float)
    parser.add_argument('--f', default=10, type=int)
    parser.add_argument('--num_rounds', default=20, type=int)
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument('--split_scheme', default='official')
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--model_arch', default=None)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--pacs_tensor_cache', default=False, action="store_true")
    parser.add_argument('--pacs_tensor_cache_dir', default=None)
    parser.add_argument('--save_checkpoints', default=False, type=str2bool)
    parser.add_argument('--disable_tqdm', default=False, type=str2bool)
    parser.add_argument('--station_parallel_gpus', default="")
    parser.add_argument('--client_method', default='ERM')
    parser.add_argument('--local_epochs', default=1, type=int)
    parser.add_argument('--n_groups_per_batch', default=2, type=int)
    parser.add_argument('--optimizer', default='torch.optim.Adam')
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--hparam1', default=1, type=float, help="irm: lambda; rex: lambda; fish: meta_lr; mixup: alpha; mmd: lambda; coral: lambda; groupdro: groupdro_eta; fedprox: mu; feddg: ratio; fedadg: alpha; fedgma: mask_threshold; fedsr: l2_regularizer;")
    parser.add_argument('--hparam2', default=1, type=float, help="fedsr: cmi_regularizer; irm: penalty_anneal_iters; fedadg: second_local_epochs")
    parser.add_argument('--hparam3', default=0, type=float)
    parser.add_argument('--hparam4', default=0, type=float)
    parser.add_argument('--hparam5', default=0, type=float)
    parser.add_argument('--activation_sketch_mode', default='diag', choices=['full', 'diag', 'blockdiag', 'lowrank', 'random_projection'])
    parser.add_argument('--activation_sketch_max_batches', default=1, type=int)
    parser.add_argument('--activation_sketch_max_patches', default=2048, type=int)
    parser.add_argument('--activation_sketch_max_full_dim', default=4096, type=int)
    parser.add_argument('--activation_sketch_block_size', default=512, type=int)
    parser.add_argument('--activation_sketch_dtype', default='float32', choices=['float32', 'float64'])
    parser.add_argument('--activation_sketch_device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--activation_sketch_shrinkage_alpha', default=0.75, type=float)
    parser.add_argument('--activation_sketch_dp_epsilon', default=-1.0, type=float)
    parser.add_argument('--activation_sketch_dp_delta', default=1e-5, type=float)
    parser.add_argument('--activation_sketch_dp_clip', default=0.0, type=float)
    parser.add_argument('--activation_sketch_lowrank_rank', default=64, type=int)
    parser.add_argument('--activation_sketch_random_projection_dim', default=256, type=int)
    parser.add_argument('--activation_sketch_random_seed', default=0, type=int)
    parser.add_argument('--align_solver', default='hungarian', choices=['hungarian', 'sinkhorn', 'greedy', 'none'])
    parser.add_argument('--align_scope', default='all')
    parser.add_argument('--graph_consistency', default=True, type=str2bool)
    parser.add_argument('--attention_head_alignment', default=True, type=str2bool)
    parser.add_argument('--residual_consistency', default=True, type=str2bool)
    parser.add_argument('--regmean_ridge', default=1e-4, type=float)
    parser.add_argument('--regmean_bias_mode', default='average', choices=['average', 'augmented'])
    parser.add_argument('--ot_solver', default='hungarian', choices=['hungarian', 'sinkhorn', 'greedy'])
    parser.add_argument('--ot_reg', default=0.05, type=float)
    parser.add_argument('--ot_iters', default=25, type=int)
    parser.add_argument('--ot_reference', default='first', choices=['first', 'largest_station'])
    parser.add_argument('--ot_scope', default='all', choices=['conv', 'linear', 'all'])
    parser.add_argument('--model_soup_type', default='uniform', choices=['uniform', 'greedy'])
    parser.add_argument('--fisher_batches', default=1, type=int)
    parser.add_argument('--fisher_eps', default=1e-8, type=float)
    parser.add_argument('--fisher_label_mode', default='true_labels', choices=['true_labels', 'predicted_labels'])
    parser.add_argument('--fisher_clip', default=0.0, type=float)
    parser.add_argument('--fisher_normalize', default=False, type=str2bool)
    parser.add_argument('--regmean_all_scope', default='all', choices=['linear', 'conv', 'attention', 'all'])
    parser.add_argument('--fedma_matching', default='hungarian', choices=['hungarian', 'greedy'])
    parser.add_argument('--fedma_scope', default='all', choices=['conv', 'linear', 'all'])
    parser.add_argument('--fedma_use_activation_signatures', default=False, type=str2bool)
    parser.add_argument('--fedrc_stat_source', default='rgb', choices=['rgb', 'features'])
    parser.add_argument('--fedrc_covariance', default='diag', choices=['diag'])
    parser.add_argument('--fedrc_tau', default=1.0, type=float)
    parser.add_argument('--fedrc_distance', default='bhattacharyya', choices=['diag_w2', 'euclidean_mean', 'kl_sym', 'bhattacharyya'])
    parser.add_argument('--fedrc_use_num_samples', default=True, type=str2bool)
    parser.add_argument('--mtgc_client_period', default=1, type=int)
    parser.add_argument('--mtgc_group_period', default=1, type=int)
    parser.add_argument('--mtgc_control_lr', default=1.0, type=float)
    parser.add_argument('--mtgc_memory_efficient', default=True, type=str2bool)
    parser.add_argument('--mtgc_allow_approx', default=False, type=str2bool)
    parser.add_argument('--fediir_penalty', default=None, type=float)
    parser.add_argument('--fediir_ema', default=0.95, type=float)
    parser.add_argument('--fediir_mean_grad_max_batches', default=None, type=int)

    args = parser.parse_args()
    main(args)

