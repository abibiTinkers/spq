import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from datasets import load_dataset

class FeatureHook:
    """Registers a forward hook to capture activations of a specific layer."""
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.activations = None

    def hook_fn(self, module, input, output):
        # Handle dict outputs common in Feature Pyramid Networks (FPN)
        if isinstance(output, dict):
            self.activations = {k: v.detach().cpu() for k, v in output.items()}
        else:
            self.activations = output.detach().cpu()

    def close(self):
        self.hook.remove()


class CalibrationFeatureExtractor:
    """Intercepts and caches FP32 activation feature maps using streamed HF data."""
    def __init__(self, target_layer_names=None):
        # REMOVED: Pre-trained weights dependency. Model initialized from scratch.
        self.model = fasterrcnn_resnet50_fpn(weights=None, pretrained_backbone=False)
        self.model.eval() # Must be in eval mode for validation/calibration
        
        # Target layers to intercept (FPN blocks essential for SPQ)
        self.target_layer_names = target_layer_names or ['backbone.fpn']
        self.hooks = {}
        self._register_hooks()

    def _register_hooks(self):
        """Finds named submodules and attaches forward activation hooks."""
        for name, module in self.model.named_modules():
            if name in self.target_layer_names:
                self.hooks[name] = FeatureHook(module)
                print(f"[Module 1] Successfully hooked layer: {name}")

    def extract_from_stream(self, hf_stream, num_samples: int = 100, batch_size: int = 4):
        """Streams images from Hugging Face, transforms them, and caches FP32 features."""
        cached_features = {name: [] for name in self.target_layer_names}
        
        # Paper Constraints: 224x224 dimensions with 3 color channels
        # REMOVED: ImageNet default mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225] constants.
        # Now uses a direct range scaling mapping to compute baseline tensor float boundaries.
        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        print(f"\n[Module 1] Starting HF Streaming Pipeline ({num_samples} samples)...")
        
        batch_tensors = []
        samples_processed = 0

        with torch.no_grad():
            # Use HF's .take() to fetch only the requested slice over the network
            for sample in hf_stream.take(num_samples):
                # 'image' key contains the raw PIL image object from HF
                raw_img = sample['image']
                
                # Convert grayscale to RGB if any outlier images exist
                if raw_img.mode != 'RGB':
                    raw_img = raw_img.convert('RGB')
                    
                # Transform and add to current mini-batch
                batch_tensors.append(img_transform(raw_img))

                # When mini-batch is full or we hit the sample limit, pass to detector
                if len(batch_tensors) == batch_size or (samples_processed + len(batch_tensors)) == num_samples:
                    # Forward pass triggers hooks automatically
                    _ = self.model(batch_tensors)
                    
                    # Store hooks data
                    for name, hook in self.hooks.items():
                        if hook.activations is not None:
                            # Clone to safely isolate tensor references from hook loop
                            if isinstance(hook.activations, dict):
                                cached_features[name].append({k: v.clone() for k, v in hook.activations.items()})
                            else:
                                cached_features[name].append(hook.activations.clone())

                    samples_processed += len(batch_tensors)
                    print(f"Streamed and Processed: {samples_processed}/{num_samples} samples")
                    batch_tensors = [] # Reset batch container

        self.cleanup()
        return cached_features

    def cleanup(self):
        """Removes all hooks from the model layers."""
        for hook in self.hooks.values():
            hook.close()
        print("[Module 1] Interceptor hooks removed safely.")


# --- Execution Block for Kaggle Notebook ---
if __name__ == "__main__":
    HF_TOKEN = "your_huggingface_read_token_here"
    DATASET_REPO = "your-hf-username/maize-leaf-disease" 

    # Initialize Streaming Dataset from HF Hub
    try:
        print("Connecting to Hugging Face Hub...")
        dataset_stream = load_dataset(
            "imagefolder", 
            data_files={"train": "**"}, 
            data_dir=DATASET_REPO,
            streaming=True,
            token=HF_TOKEN
        )
        
        train_stream = dataset_stream['train']
        
        # Instantiate Extractor and intercept 16 sample images
        extractor = CalibrationFeatureExtractor(target_layer_names=['backbone.fpn'])
        features = extractor.extract_from_stream(train_stream, num_samples=16, batch_size=4)
        
        # Verification Check
        print("\n--- Feature Map Verification ---")
        fpn_batches = features['backbone.fpn']
        print(f"Total mini-batches cached: {len(fpn_batches)}")
        
        # Check first batch layers
        for level, tensor in fpn_batches[0].items():
            print(f"FPN Level '{level}' Activation Tensor Shape: {tensor.shape}")

    except Exception as e:
        print(f"\nExecution Failed: {e}")
        print("Please check your HF_TOKEN, repository path, or ensure 'Internet On' is checked in Kaggle settings.")
