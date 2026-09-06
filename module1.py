"""
SPQ - Module 1
Calibration Feature Extraction Module

Purpose:
    1. Load ImageNet-pretrained MobileNetV2.
    2. Accept a representative calibration set.
    3. Generate FP32 intermediate activation feature maps.
    4. Cache the feature maps for Modules 2, 3 and 4.

No model training is performed.
No dataset downloading or dataset splitting is performed.
No Hugging Face token is stored here.
"""

import torch
from torchvision import transforms
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


class CalibrationFeatureExtractor:

    def __init__(
        self,
        target_layer_names=None,
        device=None
    ):

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print("Using device:", self.device)

        # ---------------------------------------------------------
        # MobileNetV2
        # ---------------------------------------------------------

        print("Loading ImageNet-pretrained MobileNetV2...")

        self.weights = MobileNet_V2_Weights.DEFAULT

        self.model = mobilenet_v2(
            weights=self.weights
        )

        self.model = self.model.float()
        self.model = self.model.to(self.device)
        self.model.eval()

        print("MobileNetV2 loaded successfully.")
        print("Precision: FP32")

        # ---------------------------------------------------------
        # Standard MobileNetV2 preprocessing
        # ---------------------------------------------------------

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ---------------------------------------------------------
        # Target layers
        # ---------------------------------------------------------

        if target_layer_names is None:
            target_layer_names = [
                "features.6",
                "features.13",
                "features.18"
            ]

        self.target_layer_names = target_layer_names

        self.target_layers = {}

        for name, layer in self.model.named_modules():

            if name in self.target_layer_names:
                self.target_layers[name] = layer

        if len(self.target_layers) != len(
            self.target_layer_names
        ):
            raise ValueError(
                "One or more target layers were not found."
            )

        print("Target layers:")

        for name in self.target_layer_names:
            print("  -", name)

        # ---------------------------------------------------------
        # Activation storage
        # ---------------------------------------------------------

        self.current_activations = {}

        self.activation_cache = {
            name: []
            for name in self.target_layer_names
        }

        # ---------------------------------------------------------
        # Register hooks
        # ---------------------------------------------------------

        self.hooks = []

        for name, layer in self.target_layers.items():

            hook = layer.register_forward_hook(
                self._create_hook(name)
            )

            self.hooks.append(hook)

        print("Forward hooks registered successfully.")

    # =============================================================
    # Forward hook
    # =============================================================

    def _create_hook(self, layer_name):

        def hook(module, inputs, output):

            if isinstance(output, torch.Tensor):

                self.current_activations[
                    layer_name
                ] = (
                    output.detach()
                    .float()
                    .cpu()
                )

        return hook

    # =============================================================
    # Preprocess one image
    # =============================================================

    def preprocess_image(self, image):

        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.preprocess(image)

    # =============================================================
    # Extract feature maps
    # =============================================================

    def extract_features(
        self,
        calibration_images,
        batch_size=32
    ):

        print("\n==============================================")
        print("SPQ MODULE 1")
        print("CALIBRATION FEATURE EXTRACTION")
        print("==============================================")

        print(
            "Calibration samples:",
            len(calibration_images)
        )

        print(
            "Batch size:",
            batch_size
        )

        # ---------------------------------------------------------
        # Reset cache
        # ---------------------------------------------------------

        self.activation_cache = {
            name: []
            for name in self.target_layer_names
        }

        # ---------------------------------------------------------
        # Process images in batches
        # ---------------------------------------------------------

        total = len(calibration_images)

        for start in range(
            0,
            total,
            batch_size
        ):

            end = min(
                start + batch_size,
                total
            )

            images = calibration_images[
                start:end
            ]

            processed_images = []

            for image in images:

                processed_images.append(
                    self.preprocess_image(image)
                )

            batch = torch.stack(
                processed_images
            )

            batch = batch.to(
                self.device,
                dtype=torch.float32
            )

            self.current_activations = {}

            # -----------------------------------------------------
            # FP32 forward pass
            # -----------------------------------------------------

            with torch.no_grad():

                self.model(batch)

            # -----------------------------------------------------
            # Store feature maps
            # -----------------------------------------------------

            for layer_name in self.target_layer_names:

                activation = (
                    self.current_activations[
                        layer_name
                    ]
                )

                self.activation_cache[
                    layer_name
                ].append(activation)

            print(
                f"Processed {end}/{total}"
            )

        # ---------------------------------------------------------
        # Verification
        # ---------------------------------------------------------

        print("\n==============================================")
        print("FEATURE EXTRACTION COMPLETED")
        print("==============================================")

        for layer_name, batches in (
            self.activation_cache.items()
        ):

            print(
                f"\nLayer: {layer_name}"
            )

            print(
                "Cached batches:",
                len(batches)
            )

            if batches:

                print(
                    "Feature-map shape:",
                    batches[0].shape
                )

                print(
                    "Dtype:",
                    batches[0].dtype
                )

        return self.activation_cache

    # =============================================================
    # Remove hooks
    # =============================================================

    def remove_hooks(self):

        for hook in self.hooks:
            hook.remove()

        self.hooks = []

        print("Forward hooks removed.")
