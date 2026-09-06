"""
SPQ - Module 1
Calibration Feature Extraction Module

Pipeline:
    Hugging Face dataset
        ↓
    Image preprocessing
        ↓
    ImageNet-pretrained MobileNetV2
        ↓
    FP32 feature-map extraction
        ↓
    Calibration activation cache

Purpose:
    Extract FP32 intermediate feature maps from a
    representative calibration subset of 1024 images.

No training is performed in this module.
No Hugging Face token is stored in this file.
"""

import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


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
        # Load ImageNet-pretrained MobileNetV2
        # ---------------------------------------------------------

        print("Loading MobileNetV2...")

        self.weights = MobileNet_V2_Weights.DEFAULT

        self.model = mobilenet_v2(
            weights=self.weights
        )

        # Keep model in FP32
        self.model = self.model.float()
        self.model = self.model.to(self.device)
        self.model.eval()

        print("MobileNetV2 loaded successfully.")
        print("Model precision: FP32")

        # ---------------------------------------------------------
        # Image preprocessing
        # ---------------------------------------------------------

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("Preprocessing initialized.")

        # ---------------------------------------------------------
        # Feature-map layers
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

        missing_layers = [
            name
            for name in self.target_layer_names
            if name not in self.target_layers
        ]

        if missing_layers:
            raise ValueError(
                "Target layers not found: "
                + str(missing_layers)
            )

        print("Target feature-map layers:")

        for name in self.target_layers:
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

        print("Forward hooks registered.")

    # =============================================================
    # Forward hook
    # =============================================================

    def _create_hook(self, layer_name):

        def hook(module, inputs, output):

            if isinstance(output, torch.Tensor):

                self.current_activations[layer_name] = (
                    output.detach()
                    .float()
                    .cpu()
                )

        return hook

    # =============================================================
    # Image preprocessing
    # =============================================================

    def preprocess_image(self, image):

        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.preprocess(image)

    # =============================================================
    # Extract calibration feature maps
    # =============================================================

    def extract_from_stream(
        self,
        dataset_stream,
        num_samples=1024,
        batch_size=32
    ):

        print("\n==============================================")
        print("SPQ MODULE 1")
        print("CALIBRATION FEATURE EXTRACTION")
        print("==============================================")

        print("Calibration images:", num_samples)
        print("Batch size:", batch_size)

        self.activation_cache = {
            name: []
            for name in self.target_layer_names
        }

        image_batch = []
        samples_processed = 0

        # ---------------------------------------------------------
        # Stream dataset
        # ---------------------------------------------------------

        for sample in dataset_stream:

            image = sample.get("image")

            if image is None:
                continue

            try:

                processed_image = self.preprocess_image(
                    image
                )

            except Exception as error:

                print(
                    "Skipping image:",
                    error
                )

                continue

            image_batch.append(processed_image)

            # -----------------------------------------------------
            # Process batch
            # -----------------------------------------------------

            if (
                len(image_batch) == batch_size
                or
                samples_processed + len(image_batch)
                >= num_samples
            ):

                remaining = (
                    num_samples
                    - samples_processed
                )

                if len(image_batch) > remaining:
                    image_batch = image_batch[:remaining]

                batch = torch.stack(
                    image_batch,
                    dim=0
                )

                batch = batch.to(
                    self.device,
                    dtype=torch.float32
                )

                self.current_activations = {}

                # -------------------------------------------------
                # FP32 forward pass
                # -------------------------------------------------

                with torch.no_grad():
                    self.model(batch)

                # -------------------------------------------------
                # Cache feature maps
                # -------------------------------------------------

                for layer_name in self.target_layer_names:

                    if layer_name not in self.current_activations:

                        raise RuntimeError(
                            "No activation captured for "
                            + layer_name
                        )

                    activation = (
                        self.current_activations[
                            layer_name
                        ]
                    )

                    self.activation_cache[
                        layer_name
                    ].append(activation)

                samples_processed += len(image_batch)

                print(
                    f"Processed "
                    f"{samples_processed}/{num_samples}"
                )

                image_batch = []

            if samples_processed >= num_samples:
                break

        # ---------------------------------------------------------
        # Verification
        # ---------------------------------------------------------

        print("\n==============================================")
        print("CALIBRATION EXTRACTION COMPLETED")
        print("==============================================")

        print(
            "Total images:",
            samples_processed
        )

        for layer_name in self.activation_cache:

            batches = self.activation_cache[layer_name]

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
                    "Feature-map dtype:",
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
