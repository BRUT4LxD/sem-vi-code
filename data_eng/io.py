import torch
import os

def save_model(model: torch.nn.Module, model_save_path: str):
    torch.save(model.state_dict(), model_save_path)


def load_model(model_instance: torch.nn.Module, model_load_path: str) -> torch.nn.Module:
    model_state_dict = torch.load(model_load_path)
    model_instance.load_state_dict(model_state_dict)
    return model_instance


def load_model_imagenette(model_path: str, model_name: str = None, device: str = 'cuda', verbose: bool = True) -> dict:
    """
    Properly load an ImageNette-trained model with full checkpoint information.
    
    This method handles the complete ImageNette model loading process including:
    - Loading the checkpoint with training metadata
    - Creating the appropriate model instance
    - Setting up the model for ImageNette (10 classes)
    - Loading the trained weights
    - Moving to the specified device
    
    Args:
        model_path (str): Path to the saved model checkpoint (.pt file)
        model_name (str, optional): Name of the model. If None, extracted from path.
        device (str): Device to load the model on ('cuda' or 'cpu')
        verbose (bool): Whether to print loading information
        
    Returns:
        dict: Dictionary containing:
            - 'model': The loaded PyTorch model
            - 'model_name': Name of the model
            - 'checkpoint': Full checkpoint data
            - 'training_state': Training metadata (if available)
            - 'device': Device the model is on
            - 'success': Boolean indicating if loading was successful
            - 'error': Error message (if loading failed)
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
        RuntimeError: If model setup or loading fails
        
    Example:
        >>> result = load_model_imagenette("./models/imagenette/resnet18_advanced.pt")
        >>> if result['success']:
        ...     model = result['model']
        ...     print(f"Model loaded: {result['model_name']}")
        ...     print(f"Best validation accuracy: {result['checkpoint']['val_accuracy']:.2f}%")
    """
    
    # Import here to avoid circular imports
    from config.imagenet_models import ImageNetModels
    from training.transfer.setup_pretraining import SetupPretraining
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📥 Loading ImageNette Model: {model_path}")
        print(f"{'='*60}")
    
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model name from path if not provided
        if model_name is None:
            from domain.model.model_names import ModelNames
            # Iterate over all model names and check if the model path contains the model name
            for model_name in ModelNames().all_model_names:
                if model_name in model_path:
                    model_name = model_name
                    break
            
            if model_name is None:
                raise ValueError(f"Model name not found in path: {model_path}")
        
        if verbose:
            print(f"📊 Checkpoint Information:")
            print(f"   Model: {model_name}")
            print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
            
            # Safe formatting for validation accuracy
            val_acc = checkpoint.get('val_accuracy', 'Unknown')
            if val_acc != 'Unknown':
                print(f"   Validation Accuracy: {val_acc:.2f}%")
            else:
                print(f"   Validation Accuracy: Unknown")
            
            # Safe formatting for validation loss
            val_loss = checkpoint.get('val_loss', 'Unknown')
            if val_loss != 'Unknown':
                print(f"   Validation Loss: {val_loss:.4f}")
            else:
                print(f"   Validation Loss: Unknown")
        
        # Create fresh model instance
        if verbose:
            print(f"🔧 Creating {model_name} model instance...")
        
        model = ImageNetModels.get_model(model_name)
        
        # Setup model for ImageNette (CRITICAL STEP!)
        if verbose:
            print(f"⚙️ Setting up model for ImageNette (10 classes)...")
        
        model = SetupPretraining.setup_imagenette(model)
        model = model.to(device)
        
        # Load the saved state
        if verbose:
            print(f"📥 Loading trained weights...")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        if verbose:
            print(f"✅ Model loaded successfully!")
        
        # Extract training state if available
        training_state = checkpoint.get('training_state', {})
        if training_state and verbose:
            print(f"📈 Training History:")
            
            # Safe formatting for best validation accuracy
            best_val_acc = training_state.get('best_val_accuracy', 'Unknown')
            if best_val_acc != 'Unknown':
                print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
            else:
                print(f"   Best Validation Accuracy: Unknown")
            
            print(f"   Best Epoch: {training_state.get('best_epoch', 'Unknown')}")
            
            # Safe formatting for total parameters
            total_params = training_state.get('total_params', 'Unknown')
            if total_params != 'Unknown':
                print(f"   Total Parameters: {total_params:,}")
            else:
                print(f"   Total Parameters: Unknown")
            
            # Safe formatting for training time
            training_time = training_state.get('total_training_time', 'Unknown')
            if training_time != 'Unknown':
                print(f"   Training Time: {training_time:.2f}s")
            else:
                print(f"   Training Time: Unknown")
        
        return {
            'model': model,
            'model_name': model_name,
            'checkpoint': checkpoint,
            'training_state': training_state,
            'device': device,
            'success': True
        }
        
    except FileNotFoundError as e:
        error_msg = f"Model file not found: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        return {
            'model': None,
            'model_name': model_name,
            'error': error_msg,
            'success': False
        }
        
    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        return {
            'model': None,
            'model_name': model_name,
            'error': error_msg,
            'success': False
        }