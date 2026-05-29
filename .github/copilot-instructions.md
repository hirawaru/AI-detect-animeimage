# AI Image Detection Project - Copilot Instructions

## Project Overview
Machine learning project for detecting AI-generated vs natural images using deep learning transfer learning.

## Key Frameworks & Libraries
- **PyTorch/Torchvision**: Deep learning framework
- **OpenCV**: Image processing
- **Scikit-learn**: Metrics and analysis
- **Jupyter**: Interactive notebooks

## Data Structure
- Raw data: `E:\archive\V5Minor100` (Synthetic + Natural folders)
- Processed: `data/train/`, `data/val/`, `data/test/`

## Key Modules
- `src/dataset.py`: Custom PyTorch Dataset class
- `src/model.py`: Model architecture wrappers
- `src/train.py`: Training loop with checkpointing
- `src/evaluate.py`: Metrics calculation
- `src/inference.py`: Prediction pipeline
- `scripts/prepare_data.py`: Data splitting utility

## Development Guidelines
1. Always use config.yaml for configuration
2. Follow PyTorch conventions for Dataset/DataLoader
3. Log metrics to tensorboard
4. Save best checkpoints based on validation loss
5. Include confusion matrix in results

## Next Steps
1. Setup Python environment and install requirements
2. Run `scripts/prepare_data.py` to split data
3. Launch `src/train.py` for training
4. Analyze results with notebooks
