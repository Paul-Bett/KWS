from src.data_loader import DataLoader

def main():
    # Initialize the data loader
    loader = DataLoader()
    
    # Download the dataset
    print("Starting dataset download...")
    train_ds, val_ds, label_names = loader.download_dataset()
    
    # Get dataset information
    info = loader.get_dataset_info()
    print("\nDataset Information:")
    print(f"Total classes: {info['total_classes']}")
    print("\nClasses available:")
    for class_name in info['classes']:
        print(f"- {class_name}")
    
    print("\nDataset verification complete!")
    print("You can now proceed with training the KWS model.")

if __name__ == "__main__":
    main() 