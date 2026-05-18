import eegimage
from config import *
from argparse import ArgumentParser

def run():
    parser = ArgumentParser(description="EEG Classification Training Script")
    parser.add_argument("--scale", type=int, default=SCALE, help="Spatial scale for input data")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--dropout_rate", type=float, default=DROPOUT_RATE, help="Dropout rate for model")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay for optimizer")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="Number of workers for data loading")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=MIN_DELTA, help="Minimum delta for early stopping")
    parser.add_argument("--device_id", type=int, default=DEVICE_ID, help="Device ID to use for training (only if CUDA is available)")
    parser.add_argument("--dual", action='store_true', help="Whether to use dual branch model with double scale and batch size")
    parser.add_argument("--double_scale", type=int, default=DOUBLE_SCALE, help="Spatial scale for double branch model")
    parser.add_argument("--double_batch_size", type=int, default=DOUBLE_BATCH_SIZE, help="Batch size for double branch model")
    
    args = parser.parse_args()
    
    if args.dual:
        args.scale = args.double_scale
        args.batch_size = args.double_batch_size

    eeg_classifier = eegimage.EEGClassifier()
    
    eeg_data = eeg_classifier.load_csv_data(FULL_PATH)
    
    train_loader, val_loader, num_classes, cw, le = eeg_classifier.data_loader(eeg_data, 
                                                                            RAW_FEATURES,
                                                                            POWER_BANDS,
                                                                            FFT_FEATURE_COLUMNS,
                                                                            batch_size=args.batch_size,
                                                                            scale=args.scale,
                                                                            num_workers=args.num_workers)
    
    model, optimizer, criterion, scheduler = eeg_classifier.build_model(input_channels=INPUT_CHANNELS,
                                                                        spatial_size=args.scale,
                                                                        num_classes=num_classes,
                                                                        cw=cw,
                                                                        learning_rate=args.learning_rate,
                                                                        weight_decay=args.weight_decay,
                                                                        dropout_rate=args.dropout_rate,
                                                                        config_path=CONFIG_PATH,
                                                                        arch_path=ARCH_PATH)
    
    eeg_model = eeg_classifier.train(train_loader,
                        val_loader,
                        args.epochs,
                        args.learning_rate,
                        args.weight_decay,
                        device=DEVICE)
    
    eeg_classifier.evaluate(eeg_model, 
                            val_loader,
                            device=DEVICE,
                            le=le)

if __name__ == "__main__":
    run()