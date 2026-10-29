import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.tuning import (GridSearchCV, RandomSearchCV, BayesianOptimization,
                        PARAM_GRID, PARAM_GRID_SMALL, PARAM_GRID_FINE)
from src.utils import set_seed, setup_logger, save_json

def parse_args():
    parser = argparse.ArgumentParser(description='SAG Hyperparameter Search')
    
    parser.add_argument('--dataset', type=str, default='BlogCatalog',
                        choices=['BlogCatalog', 'Flickr', 'ACM', 'Cora',
                                'Citeseer', 'Pubmed', 'Amazon', 'Enron'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    parser.add_argument('--method', type=str, default='grid',
                        choices=['grid', 'random', 'bayesian'])
    parser.add_argument('--grid_size', type=str, default='small',
                        choices=['small', 'default', 'fine'])
    
    parser.add_argument('--n_iter', type=int, default=30,
                        help='Number of iterations for random/bayesian search')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of runs per parameter setting')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs per run')
    parser.add_argument('--metric', type=str, default='auroc',
                        choices=['auroc', 'auprc', 'macro_f1'])
    
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.seed = args.seed
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'{args.dataset}_tuning_{args.method}_{timestamp}'
    
    result_dir = os.path.join(args.output_dir, 'tuning', exp_name)
    log_dir = os.path.join(result_dir, 'logs')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logger('SAG_Tuning', log_dir)
    logger.info(f"Starting hyperparameter search on {args.dataset}")
    logger.info(f"Method: {args.method}")
    set_seed(Config.seed)
    
    logger.info("Loading data...")
    dataset_config = get_dataset_config(args.dataset)
    data_loader = DataLoader(
        data_dir=Config.data_dir,
        dataset_name=args.dataset,
        clique_size=dataset_config.get('clique_size', 15),
        num_cliques=dataset_config.get('num_cliques', 10),
        seed=Config.seed
    )
    data_loader.load()
    logger.info(f"Dataset stats: {data_loader.get_stats()}")
    
    if args.method == 'grid':
        if args.grid_size == 'small':
            param_grid = PARAM_GRID_SMALL
        elif args.grid_size == 'fine':
            param_grid = PARAM_GRID_FINE
        else:
            param_grid = PARAM_GRID
        
        logger.info(f"Grid search with {args.grid_size} grid")
        logger.info(f"Parameter grid: {param_grid}")
        
        searcher = GridSearchCV(
            param_grid=param_grid,
            data_loader=data_loader,
            config=Config,
            n_runs=args.n_runs,
            num_epochs=args.num_epochs,
            metric=args.metric,
            logger=logger
        )
    
    elif args.method == 'random':
        param_distributions = {
            'gamma': (0.85, 0.99),
            'beta_upd': (0.3, 3.0),
            'alpha': (0.3, 3.0),
            'delta': (0.3, 3.0),
            'rho': (1.2, 4.0),
        }
        
        logger.info(f"Random search with {args.n_iter} iterations")
        logger.info(f"Parameter distributions: {param_distributions}")
        
        searcher = RandomSearchCV(
            param_distributions=param_distributions,
            data_loader=data_loader,
            config=Config,
            n_iter=args.n_iter,
            n_runs=args.n_runs,
            num_epochs=args.num_epochs,
            metric=args.metric,
            logger=logger,
            seed=args.seed
        )
    
    elif args.method == 'bayesian':
        param_bounds = {
            'gamma': (0.85, 0.99),
            'beta_upd': (0.3, 3.0),
            'alpha': (0.3, 3.0),
            'delta': (0.3, 3.0),
            'rho': (1.2, 4.0),
        }
        
        logger.info(f"Bayesian optimization with {args.n_iter} iterations")
        logger.info(f"Parameter bounds: {param_bounds}")
        
        searcher = BayesianOptimization(
            param_bounds=param_bounds,
            data_loader=data_loader,
            config=Config,
            n_iter=args.n_iter,
            n_runs=args.n_runs,
            num_epochs=args.num_epochs,
            metric=args.metric,
            logger=logger,
            seed=args.seed
        )
    
    logger.info("\nStarting search...")
    best_params, best_score = searcher.search()
    
    logger.info("\n" + "=" * 60)
    logger.info("SEARCH COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Best {args.metric}: {best_score:.4f}")
    logger.info(f"Best parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    results_path = os.path.join(result_dir, 'search_results.json')
    searcher.save_results(results_path)
    logger.info(f"\nResults saved to {results_path}")
    
    best_config_path = os.path.join(result_dir, 'best_config.json')
    save_json({
        'dataset': args.dataset,
        'method': args.method,
        'metric': args.metric,
        'best_score': best_score,
        'best_params': best_params,
        'timestamp': timestamp
    }, best_config_path)
    
    if args.method == 'grid':
        try:
            df = searcher.get_results_df()
            csv_path = os.path.join(result_dir, 'grid_search_results.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Grid search results saved to {csv_path}")
            
            logger.info("\nTop 10 parameter combinations:")
            logger.info(df.head(10).to_string())
        except ImportError:
            logger.info("pandas not available, skipping CSV export")
    
    logger.info("\nHyperparameter search completed!")
    
    return best_params, best_score

if __name__ == '__main__':
    main()
