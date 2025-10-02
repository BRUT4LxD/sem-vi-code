from datetime import datetime
from data_eng.io import save_model
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from evaluation.validation import Validation, ValidationAccuracyResult
from training.transfer.setup_pretraining import SetupPretraining

from torch.nn import Module, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm

class Training:

    @staticmethod
    def simple_train(
        model: Module,
        loss_fn,
        optimizer: Adam,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        num_epochs: int = 5,
        device: str= 'cuda',
        save_model_path: str = None,
        model_name: str = None,
        writer: SummaryWriter = None,
        is_binary_classification: bool = False,
        stop_at_loss: float = None):

        n_total_steps = len(train_loader)
        stop_at_loss = stop_at_loss if stop_at_loss is not None else 0.00000001
        scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
        model_name = model_name if model_name is not None else model.__class__.__name__
        for epoch in range(num_epochs):
            i = 0
            for images, labels in tqdm(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                if is_binary_classification:
                    outputs = outputs.squeeze(1)
                    labels = labels.float()

                loss: torch.Tensor = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

                if (i + 1) % 2000 == 0:
                    print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.8f}')

                i = i + 1

            scheduler.step()
            print(f'({model_name}) epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.8f} lr = {scheduler.get_last_lr()[0]}')
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

            if test_loader is not None:
                res: ValidationAccuracyResult

                if is_binary_classification:
                    res = Validation.validate_binary_classification(model, test_loader, device, False, model_name)
                else:
                    res = Validation.validate_imagenet_with_imagenette_classes(
                    model=model,
                    test_loader=test_loader,
                    model_name=model_name,
                    device=device,
                    print_results=False
                )
                writer.add_scalar("Accuracy/train", res.accuracy, epoch)
                _ = model.train()

            if loss.item() < stop_at_loss:
                break

        print('Finished training')

        if save_model_path is not None:
            save_model(model, save_model_path)

    @staticmethod
    def train_imagenette(
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        device: str = 'cuda',
        save_model_path: str = None,
        model_name: str = None,
        writer: SummaryWriter = None,
        setup_model: bool = True,
        validation_frequency: int = 1,
        early_stopping_patience: int = 5,
        min_delta: float = 0.001,
        scheduler_type: str = 'step',
        scheduler_params: dict = None,
        gradient_clip_norm: float = None,
        weight_decay: float = 0.0,
        verbose: bool = True) -> dict:
        """
        Robust ImageNette training method with comprehensive observability and error handling.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            test_loader: Validation data loader (optional)
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            device: Device to train on ('cuda' or 'cpu')
            save_model_path: Path to save the trained model
            model_name: Name of the model for logging
            writer: TensorBoard writer for logging
            setup_model: Whether to setup model for ImageNette (10 classes)
            validation_frequency: How often to run validation (every N epochs)
            early_stopping_patience: Number of epochs to wait before early stopping
            min_delta: Minimum change to qualify as improvement
            scheduler_type: Type of learning rate scheduler ('step', 'plateau', 'cosine')
            scheduler_params: Parameters for the scheduler
            gradient_clip_norm: Gradient clipping norm (None to disable)
            weight_decay: L2 regularization weight
            verbose: Whether to print training progress
            
        Returns:
            dict: Training results including metrics and model state
        """
        
        # Input validation
        if not isinstance(model, Module):
            raise TypeError(f"Model must be a PyTorch Module, got {type(model)}")
        
        if train_loader is None:
            raise ValueError("train_loader cannot be None")
        
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        
        # Setup device
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f"🚀 Training on device: {device}")
        
        # Setup model name
        if model_name is None:
            model_name = model.__class__.__name__
        
        # Setup TensorBoard logging
        if writer is None and verbose:
            date = datetime.now().strftime("%d-%m-%Y_%H-%M")
            log_dir = f'runs/imagenette_training/{model_name}/{date}_lr={learning_rate}'
            writer = SummaryWriter(log_dir=log_dir)
            if verbose:
                print(f"📊 TensorBoard logging to: {log_dir}")
        
        # Setup model for ImageNette if requested
        if setup_model:
            if verbose:
                print(f"⚙️ Setting up {model_name} for ImageNette training...")
            try:
                model = SetupPretraining.setup_imagenette(model)
                if verbose:
                    print(f"✅ Model setup complete - ready for ImageNette (10 classes)")
            except Exception as e:
                raise RuntimeError(f"Failed to setup model for ImageNette: {str(e)}")
        
        # Move model to device
        model = model.to(device)
        model.train()
        
        # Setup loss function and optimizer
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup learning rate scheduler
        scheduler = Training._setup_scheduler(
            optimizer, scheduler_type, scheduler_params, num_epochs
        )
        
        # Training state tracking
        training_state = {
            'best_val_accuracy': 0.0,
            'best_epoch': 0,
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'learning_rates': [],
            'epoch_times': [],
            'early_stopping_counter': 0,
            'model_name': model_name,
            'device': str(device),
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        if verbose:
            print(f"📈 Training Configuration:")
            print(f"   Model: {model_name}")
            print(f"   Parameters: {training_state['trainable_params']:,} trainable / {training_state['total_params']:,} total")
            print(f"   Learning Rate: {learning_rate}")
            print(f"   Epochs: {num_epochs}")
            print(f"   Batch Size: {train_loader.batch_size}")
            print(f"   Scheduler: {scheduler_type}")
            print(f"   Early Stopping: {early_stopping_patience} epochs")
        
        # Training loop
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            epoch_start_time = datetime.now()
            
            # Training phase
            train_metrics = Training._train_epoch(
                model, train_loader, criterion, optimizer, device, 
                gradient_clip_norm, verbose, epoch, num_epochs
            )
            
            # Validation phase
            val_metrics = None
            if test_loader is not None and (epoch + 1) % validation_frequency == 0:
                val_metrics = Validation.validate_imagenette_epoch(
                    model, test_loader, criterion, device, verbose, epoch, num_epochs
                )
            
            # Update learning rate
            if scheduler is not None:
                if scheduler_type == 'plateau' and val_metrics is not None:
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Record metrics
            current_lr = optimizer.param_groups[0]['lr']
            training_state['learning_rates'].append(current_lr)
            training_state['train_losses'].append(train_metrics['loss'])
            
            if val_metrics is not None:
                training_state['val_losses'].append(val_metrics['loss'])
                training_state['val_accuracies'].append(val_metrics['accuracy'])
                
                # Check for best model
                if val_metrics['accuracy'] > training_state['best_val_accuracy']:
                    training_state['best_val_accuracy'] = val_metrics['accuracy']
                    training_state['best_epoch'] = epoch + 1
                    training_state['early_stopping_counter'] = 0
                    
                    # Save best model
                    if save_model_path is not None:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch + 1,
                            'val_accuracy': val_metrics['accuracy'],
                            'val_loss': val_metrics['loss'],
                            'training_state': training_state
                        }, save_model_path)
                        if verbose:
                            print(f"💾 Best model saved: {save_model_path}")
                else:
                    training_state['early_stopping_counter'] += 1
            else:
                training_state['val_losses'].append(None)
                training_state['val_accuracies'].append(None)
            
            # Logging
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            training_state['epoch_times'].append(epoch_time)
            
            if writer is not None:
                writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
                writer.add_scalar("Learning_Rate", current_lr, epoch)
                writer.add_scalar("Epoch_Time", epoch_time, epoch)
                
                if val_metrics is not None:
                    writer.add_scalar("Loss/Validation", val_metrics['loss'], epoch)
                    writer.add_scalar("Accuracy/Validation", val_metrics['accuracy'], epoch)
            
            # Early stopping check
            if (val_metrics is not None and 
                training_state['early_stopping_counter'] >= early_stopping_patience):
                if verbose:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Progress summary
            if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
                Training._print_epoch_summary(
                    epoch + 1, num_epochs, train_metrics, val_metrics, 
                    current_lr, epoch_time, training_state
                )
        
        # Final summary
        total_time = (datetime.now() - start_time).total_seconds()
        training_state['total_training_time'] = total_time
        
        if verbose:
            Training._print_final_summary(training_state, save_model_path)
        
        # Close writer
        if writer is not None:
            writer.close()
        
        return training_state

    @staticmethod
    def _setup_scheduler(optimizer, scheduler_type, scheduler_params, num_epochs):
        """Setup learning rate scheduler."""
        if scheduler_params is None:
            scheduler_params = {}
        
        if scheduler_type == 'step':
            return StepLR(optimizer, step_size=scheduler_params.get('step_size', 5), 
                         gamma=scheduler_params.get('gamma', 0.8))
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='max', factor=scheduler_params.get('factor', 0.5),
                                   patience=scheduler_params.get('patience', 3), verbose=True)
        elif scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=num_epochs, 
                                   eta_min=scheduler_params.get('eta_min', 0))
        else:
            return None

    @staticmethod
    def _train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip_norm, verbose, epoch, num_epochs):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                          disable=not verbose)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_predictions += pred.eq(target).sum().item()
            total_samples += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct_predictions / total_samples:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct_predictions / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}


    @staticmethod
    def _print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics, lr, epoch_time, training_state):
        """Print epoch summary."""
        print(f"\n📊 Epoch {epoch}/{num_epochs} Summary:")
        print(f"   Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        
        if val_metrics is not None:
            print(f"   Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"   Best Val Acc: {training_state['best_val_accuracy']:.2f}% (Epoch {training_state['best_epoch']})")
        
        print(f"   Learning Rate: {lr:.6f}")
        print(f"   Epoch Time: {epoch_time:.2f}s")

    @staticmethod
    def _print_final_summary(training_state, save_model_path):
        """Print final training summary."""
        print(f"\n🎉 Training Completed!")
        print(f"{'='*50}")
        print(f"Model: {training_state['model_name']}")
        print(f"Device: {training_state['device']}")
        print(f"Total Parameters: {training_state['total_params']:,}")
        print(f"Trainable Parameters: {training_state['trainable_params']:,}")
        print(f"Training Time: {training_state['total_training_time']:.2f}s")
        
        if training_state['val_accuracies']:
            print(f"Best Validation Accuracy: {training_state['best_val_accuracy']:.2f}% (Epoch {training_state['best_epoch']})")
            print(f"Final Validation Accuracy: {training_state['val_accuracies'][-1]:.2f}%")
        
        print(f"Final Training Loss: {training_state['train_losses'][-1]:.4f}")
        
        if save_model_path:
            print(f"Model saved to: {save_model_path}")
        
        print(f"{'='*50}")

    @staticmethod
    def train_binary(
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        learning_rate=0.00001,
        num_epochs=5,
        device='cuda',
        save_model_path=None,
        model_name=None,
        writer: SummaryWriter = None):

        if writer is None:
            date = datetime.now().strftime("%d-%m-%Y_%H-%M")
            writer = SummaryWriter(log_dir=f'runs/binary_training/{model_name}/{date}_lr={learning_rate}')

        if save_model_path is None:
            save_model_path = f'./models/binary/{model_name}.pt'

        _ = model.train()
        _ = model.to(device)
        criterion = BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        Training.simple_train(
            model=model,
            loss_fn=criterion,
            optimizer = optimizer,
            train_loader=train_loader,test_loader=test_loader,
            num_epochs=num_epochs,
            device=device,
            save_model_path=save_model_path,
            model_name=model_name,
            writer=writer,
            is_binary_classification=True,
            stop_at_loss=0.0000000001)

    @staticmethod
    def train_imagenette_noise_detection(
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        device: str = 'cuda',
        save_model_path: str = None,
        model_name: str = None,
        writer: SummaryWriter = None,
        setup_model: bool = True,
        validation_frequency: int = 1,
        early_stopping_patience: int = 7,
        min_delta: float = 0.001,
        scheduler_type: str = 'plateau',
        scheduler_params: dict = None,
        gradient_clip_norm: float = None,
        weight_decay: float = 0.0001,
        verbose: bool = True) -> dict:
        """
        Train a binary classifier to detect adversarial noise in ImageNette images.
        
        This method is specifically designed for training models to distinguish between
        clean and adversarial images using the attacked ImageNette dataset.
        
        Args:
            model: PyTorch model to train (will be setup for binary classification)
            train_loader: Training data loader with binary labels (0=clean, 1=adversarial)
            test_loader: Validation data loader with binary labels
            learning_rate: Learning rate for optimizer (default: 0.001)
            num_epochs: Number of training epochs (default: 20)
            device: Device to train on ('cuda' or 'cpu')
            save_model_path: Path to save the trained model
            model_name: Name of the model for logging
            writer: TensorBoard writer for logging
            setup_model: Whether to setup model for binary classification (default: True)
            validation_frequency: How often to run validation (every N epochs)
            early_stopping_patience: Number of epochs to wait before early stopping
            min_delta: Minimum change to qualify as improvement
            scheduler_type: Type of learning rate scheduler ('step', 'plateau', 'cosine')
            scheduler_params: Parameters for the scheduler
            gradient_clip_norm: Gradient clipping norm (None to disable)
            weight_decay: L2 regularization weight (default: 0.0001)
            verbose: Whether to print training progress
            
        Returns:
            dict: Training results including metrics, model state, and performance statistics
        """
        
        # Input validation
        if not isinstance(model, Module):
            raise TypeError(f"Model must be a PyTorch Module, got {type(model)}")
        
        if train_loader is None:
            raise ValueError("train_loader cannot be None")
        
        if test_loader is None:
            raise ValueError("test_loader cannot be None for noise detection training")
        
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        
        # Setup device
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f"🚀 Adversarial Noise Detection Training")
            print(f"{'='*60}")
            print(f"Device: {device}")
        
        # Setup model name
        if model_name is None:
            model_name = model.__class__.__name__
        
        # Setup TensorBoard logging
        if writer is None:
            date = datetime.now().strftime("%d-%m-%Y_%H-%M")
            log_dir = f'runs/binary_training/{model_name}/{date}_lr={learning_rate}'
            writer = SummaryWriter(log_dir=log_dir)
            if verbose:
                print(f"📊 TensorBoard logging to: {log_dir}")
        
        # Setup model for binary classification if requested
        if setup_model:
            if verbose:
                print(f"⚙️ Setting up {model_name} for binary classification...")
            try:
                model = SetupPretraining.setup_binary(model)
                if verbose:
                    print(f"✅ Model setup complete - ready for binary classification (1 output)")
            except Exception as e:
                raise RuntimeError(f"Failed to setup model for binary classification: {str(e)}")
        
        # Move model to device
        model = model.to(device)
        model.train()
        
        # Setup loss function and optimizer
        criterion = BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup learning rate scheduler
        scheduler = Training._setup_scheduler(
            optimizer, scheduler_type, scheduler_params, num_epochs
        )
        
        # Training state tracking
        training_state = {
            'best_val_accuracy': 0.0,
            'best_epoch': 0,
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'learning_rates': [],
            'epoch_times': [],
            'early_stopping_counter': 0,
            'model_name': model_name,
            'device': str(device),
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'best_precision': 0.0,
            'best_recall': 0.0,
            'best_f1': 0.0
        }
        
        if verbose:
            print(f"📈 Training Configuration:")
            print(f"   Model: {model_name}")
            print(f"   Parameters: {training_state['trainable_params']:,} trainable / {training_state['total_params']:,} total")
            print(f"   Learning Rate: {learning_rate}")
            print(f"   Weight Decay: {weight_decay}")
            print(f"   Epochs: {num_epochs}")
            print(f"   Batch Size: {train_loader.batch_size}")
            print(f"   Scheduler: {scheduler_type}")
            print(f"   Early Stopping: {early_stopping_patience} epochs")
            print(f"   Loss Function: BCEWithLogitsLoss")
        
        # Training loop
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            epoch_start_time = datetime.now()
            
            # Training phase
            model.train()
            total_train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                              disable=not verbose)
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                data = data.to(device)
                target = target.to(device).float().unsqueeze(1)  # (batch_size, 1)
                
                optimizer.zero_grad()
                output = model(data)  # (batch_size, 1)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                
                optimizer.step()
                
                # Update metrics
                total_train_loss += loss.item()
                predicted = (torch.sigmoid(output) > 0.5).float()
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * train_correct / train_total:.2f}%'
                })
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = 100. * train_correct / train_total
            
            # Validation phase
            val_metrics = None
            if (epoch + 1) % validation_frequency == 0:
                model.eval()
                total_val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                # For binary classification metrics
                true_positives = 0
                false_positives = 0
                true_negatives = 0
                false_negatives = 0
                
                with torch.no_grad():
                    for data, target in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                                            disable=not verbose, leave=False):
                        data = data.to(device)
                        target = target.to(device).float().unsqueeze(1)
                        
                        output = model(data)
                        loss = criterion(output, target)
                        
                        total_val_loss += loss.item()
                        predicted = (torch.sigmoid(output) > 0.5).float()
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                        
                        # Calculate confusion matrix components
                        true_positives += ((predicted == 1) & (target == 1)).sum().item()
                        false_positives += ((predicted == 1) & (target == 0)).sum().item()
                        true_negatives += ((predicted == 0) & (target == 0)).sum().item()
                        false_negatives += ((predicted == 0) & (target == 1)).sum().item()
                
                avg_val_loss = total_val_loss / len(test_loader)
                val_accuracy = 100. * val_correct / val_total
                
                # Calculate precision, recall, F1
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                val_metrics = {
                    'loss': avg_val_loss,
                    'accuracy': val_accuracy,
                    'precision': precision * 100,
                    'recall': recall * 100,
                    'f1': f1 * 100,
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'true_negatives': true_negatives,
                    'false_negatives': false_negatives
                }
            
            # Update learning rate
            if scheduler is not None and val_metrics is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(val_metrics['accuracy'])
                else:
                    scheduler.step()
            
            # Record metrics
            current_lr = optimizer.param_groups[0]['lr']
            training_state['learning_rates'].append(current_lr)
            training_state['train_losses'].append(avg_train_loss)
            
            if val_metrics is not None:
                training_state['val_losses'].append(val_metrics['loss'])
                training_state['val_accuracies'].append(val_metrics['accuracy'])
                
                # Check for best model
                if val_metrics['accuracy'] > training_state['best_val_accuracy'] + min_delta:
                    training_state['best_val_accuracy'] = val_metrics['accuracy']
                    training_state['best_epoch'] = epoch + 1
                    training_state['best_precision'] = val_metrics['precision']
                    training_state['best_recall'] = val_metrics['recall']
                    training_state['best_f1'] = val_metrics['f1']
                    training_state['early_stopping_counter'] = 0
                    
                    # Save best model
                    if save_model_path is not None:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch + 1,
                            'val_accuracy': val_metrics['accuracy'],
                            'val_loss': val_metrics['loss'],
                            'precision': val_metrics['precision'],
                            'recall': val_metrics['recall'],
                            'f1': val_metrics['f1'],
                            'training_state': training_state
                        }, save_model_path)
                        if verbose:
                            print(f"💾 Best model saved: {save_model_path}")
                else:
                    training_state['early_stopping_counter'] += 1
            
            # Logging
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            training_state['epoch_times'].append(epoch_time)
            
            if writer is not None:
                writer.add_scalar("Loss/Train", avg_train_loss, epoch)
                writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
                writer.add_scalar("Learning_Rate", current_lr, epoch)
                writer.add_scalar("Epoch_Time", epoch_time, epoch)
                
                if val_metrics is not None:
                    writer.add_scalar("Loss/Validation", val_metrics['loss'], epoch)
                    writer.add_scalar("Accuracy/Validation", val_metrics['accuracy'], epoch)
                    writer.add_scalar("Precision/Validation", val_metrics['precision'], epoch)
                    writer.add_scalar("Recall/Validation", val_metrics['recall'], epoch)
                    writer.add_scalar("F1/Validation", val_metrics['f1'], epoch)
            
            # Print epoch summary
            if verbose:
                print(f"\n📊 Epoch {epoch+1}/{num_epochs} Summary:")
                print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
                if val_metrics is not None:
                    print(f"   Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
                    print(f"   Precision: {val_metrics['precision']:.2f}% | Recall: {val_metrics['recall']:.2f}% | F1: {val_metrics['f1']:.2f}%")
                    print(f"   TP: {val_metrics['true_positives']} | FP: {val_metrics['false_positives']} | "
                          f"TN: {val_metrics['true_negatives']} | FN: {val_metrics['false_negatives']}")
                    print(f"   Best Val Acc: {training_state['best_val_accuracy']:.2f}% (Epoch {training_state['best_epoch']})")
                print(f"   Learning Rate: {current_lr:.6f}")
                print(f"   Epoch Time: {epoch_time:.2f}s")
            
            # Early stopping check
            if training_state['early_stopping_counter'] >= early_stopping_patience:
                if verbose:
                    print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                    print(f"   No improvement for {early_stopping_patience} consecutive epochs")
                break
        
        # Final summary
        total_time = (datetime.now() - start_time).total_seconds()
        training_state['total_training_time'] = total_time
        
        if verbose:
            print(f"\n🎉 Noise Detection Training Completed!")
            print(f"{'='*60}")
            print(f"Model: {training_state['model_name']}")
            print(f"Device: {training_state['device']}")
            print(f"Total Parameters: {training_state['total_params']:,}")
            print(f"Trainable Parameters: {training_state['trainable_params']:,}")
            print(f"Training Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            print(f"\n📊 Best Performance:")
            print(f"   Validation Accuracy: {training_state['best_val_accuracy']:.2f}% (Epoch {training_state['best_epoch']})")
            print(f"   Precision: {training_state['best_precision']:.2f}%")
            print(f"   Recall: {training_state['best_recall']:.2f}%")
            print(f"   F1 Score: {training_state['best_f1']:.2f}%")
            
            if save_model_path:
                print(f"\n💾 Model saved to: {save_model_path}")
            print(f"{'='*60}")
        
        # Close writer
        if writer is not None:
            writer.close()
        
        return training_state