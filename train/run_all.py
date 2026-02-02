import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from train_vehicle_estimation import run_training as train_estimation
from train_vehicle_predict import run_training as train_prediction


ESTIMATION_CONFIG = {
    'num_samples': 2000,
    'max_vehicles': 50,
    'batch_size': 16,
    'num_workers': 0,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'num_epochs': 100,
    'save_dir': 'checkpoints/estimation',
    'data_dir': None,
    'resume_from': None
}

PREDICTION_CONFIG = {
    'in_channels': 1,
    'num_zones': 8,
    'embed_dim': 64,
    'spatial_size': 18,
    'num_encoder_layers': 4,
    'num_heads': 4,
    'mlp_ratio': 4,
    'dropout': 0.1,
    'prediction_horizons': [1, 3, 5, 10, 15],
    'max_seq_len': 50,
    'seq_len': 24,
    'lite': False,
    'num_samples': 2000,
    'batch_size': 16,
    'num_workers': 0,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'num_epochs': 100,
    'save_dir': 'checkpoints/prediction',
    'density_dir': None,
    'resume_from': None
}


def main():
    parser = argparse.ArgumentParser(description='Train traffic signal models')
    parser.add_argument('--step', type=int, choices=[1, 2], required=True,
                        help='Training step: 1=estimation, 2=prediction')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--lite', action='store_true', help='Use lite model for prediction')
    
    args = parser.parse_args()
    
    if args.step == 1:
        config = ESTIMATION_CONFIG.copy()
        config['num_epochs'] = args.epochs
        config['batch_size'] = args.batch_size
        config['lr'] = args.lr
        if args.data_dir:
            config['data_dir'] = Path(args.data_dir)
        if args.resume:
            config['resume_from'] = args.resume
            
        print("=" * 50)
        print("Training Step 1: Vehicle Estimation (Image -> Density)")
        print("=" * 50)
        train_estimation(config)
        
    elif args.step == 2:
        config = PREDICTION_CONFIG.copy()
        config['num_epochs'] = args.epochs
        config['batch_size'] = args.batch_size
        config['lr'] = args.lr
        config['lite'] = args.lite
        if args.data_dir:
            config['density_dir'] = args.data_dir
        if args.resume:
            config['resume_from'] = args.resume
            
        print("=" * 50)
        print("Training Step 2: Vehicle Prediction (Density -> Future Counts)")
        print("=" * 50)
        train_prediction(config)


if __name__ == '__main__':
    main()
