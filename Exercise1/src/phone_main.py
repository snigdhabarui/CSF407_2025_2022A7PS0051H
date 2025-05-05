import os
import argparse
from PhoneNum_trainer import PhoneNumberTrainer
from PhoneNum_data import get_data_loaders

def main():
    parser = argparse.ArgumentParser(description="Train Phone Number Recognition Model")
    parser.add_argument('--config', type=str, default='phone_config.json')
    parser.add_argument('--dataset_type', type=str, choices=['balanced', 'imbalanced'], default='balanced')
    args = parser.parse_args()

    config_path = args.config
    dataset_type = args.dataset_type

    data_loaders, balanced_path, imbalanced_path = get_data_loaders(
        create_new=True,
        batch_size=64,
        train_ratio=0.7,
        val_ratio=0.15,
        num_samples=20000,
        save_dir='./Dataset/'
    )

    trainer = PhoneNumberTrainer(config_path)

    trainer.train(data_loaders[dataset_type])

    test_metrics = trainer.test(data_loaders[dataset_type]['test'])

    cross_metrics = trainer.cross_dataset_evaluation(data_loaders)

    print("Training and evaluation complete!")
    return test_metrics, cross_metrics

if __name__ == "__main__":
    main()
