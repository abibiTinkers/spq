"""
SPQ - Module 1
Calibration Feature Extraction Module

Purpose:
    1. Load a pretrained FP32 MobileNetV2 model.
    2. Stream a representative calibration subset from Hugging Face.
    3. Apply model-compatible preprocessing.
    4. Extract FP32 activation feature maps.
    5. Cache the activation tensors for subsequent SPQ modules.

The Hugging Face token is NOT stored in this file.
"""

import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


class CalibrationFeatureExtractor:
    """
    Module 1 of the SPQ pipeline.

    This class:
        - initializes a pretrained FP32 model
        - preprocesses streamed images
        - extracts intermediate FP32 activations
        - stores them in a calibration cache
    """

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

        print(f"Using device: {self.device}")

        # ---------------------------------------------------------
        # Load pretrained MobileNetV2
        # ---------------------------------------------------------

        print("Loading pretrained MobileNetV2...")

        self.weights = MobileNet_V2_Weights.DEFAULT

        self.model = mobilenet_v2(
            weights=self.weights
        )

        # Important:
        # Keep the model in FP32.
        self.model = self.model.float()

        self.model.to(self.device)

        self.model.eval()

        print("Pretrained FP32 MobileNetV2 loaded successfully.")

        # ---------------------------------------------------------
        # Model-compatible preprocessing
        # ---------------------------------------------------------

        self.preprocess = self.weights.transforms()

        print("Preprocessing pipeline initialized.")

        # ---------------------------------------------------------
        # Default activation layers
        # ---------------------------------------------------------

        if target_layer_names is None:

            target_layer_names = [
                "features.6",
                "features.13",
                "features.18"
            ]

        self.target_layer_names = target_layer_names

        # ---------------------------------------------------------
        # Find requested layers
        # ---------------------------------------------------------

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
                "The following target layers were not found: "
                + str(missing_layers)
            )

        print("Target activation layers:")

        for name in self.target_layers:
            print(f"  - {name}")

        # ---------------------------------------------------------
        # Activation storage
        # ---------------------------------------------------------

        self.current_activations = {}

        self.activation_cache = {
            name: []
            for name in self.target_layer_names
        }

        # ---------------------------------------------------------
        # Register forward hooks
        # ---------------------------------------------------------

        self.hooks = []

        for name, layer in self.target_layers.items():

            hook = layer.register_forward_hook(
                self._create_hook(name)
            )

            self.hooks.append(hook)

        print("Forward hooks registered successfully.")

    # =============================================================
    # Forward Hook
    # =============================================================

    def _create_hook(self, layer_name):

        def hook(module, inputs, output):

            # Detach from computation graph.
            #
            # Move to CPU so that the calibration cache
            # does not unnecessarily occupy GPU memory.

            if isinstance(output, torch.Tensor):

                self.current_activations[layer_name] = (
                    output.detach()
                    .float()
                    .cpu()
                )

            elif isinstance(output, (tuple, list)):

                self.current_activations[layer_name] = (
                    output[0]
                    .detach()
                    .float()
                    .cpu()
                )

        return hook

    # =============================================================
    # Image preprocessing
    # =============================================================

    def preprocess_image(self, image):

        """
        Apply the preprocessing associated with the
        pretrained MobileNetV2 weights.
        """

        return self.preprocess(image)

    # =============================================================
    # Extract calibration features
    # =============================================================

    def extract_from_stream(
        self,
        dataset_stream,
        num_samples=1024,
        batch_size=32
    ):

        """
        Extract FP32 activation feature maps from a streamed
        calibration subset.

        Parameters
        ----------
        dataset_stream:
            Hugging Face streaming dataset.

        num_samples:
            Number of calibration images.

        batch_size:
            Number of images processed at a time.

        Returns
        -------
        dict:
            Dictionary containing FP32 activation tensors.
        """

        print("\n==============================================")
        print("SPQ MODULE 1")
        print("Calibration Feature Extraction")
        print("==============================================")

        print(f"Calibration samples: {num_samples}")
        print(f"Batch size: {batch_size}")

        # ---------------------------------------------------------
        # Reset cache
        # ---------------------------------------------------------

        self.activation_cache = {
            name: []
            for name in self.target_layer_names
        }

        # ---------------------------------------------------------
        # Temporary image batch
        # ---------------------------------------------------------

        image_batch = []

        samples_processed = 0

        # ---------------------------------------------------------
        # Stream images from Hugging Face
        # ---------------------------------------------------------

        for sample in dataset_stream:

            # -----------------------------------------------------
            # Obtain image
            # -----------------------------------------------------

            image = sample.get("image")

            if image is None:
                continue

            # -----------------------------------------------------
            # Apply preprocessing
            # -----------------------------------------------------

            try:

                processed_image = self.preprocess_image(
                    image
                )

            except Exception as error:

                print(
                    f"Skipping image because preprocessing failed: "
                    f"{error}"
                )

                continue

            image_batch.append(processed_image)

            # -----------------------------------------------------
            # Stop after requested calibration samples
            # -----------------------------------------------------

            if (
                len(image_batch) == batch_size
                or
                samples_processed + len(image_batch)
                >= num_samples
            ):

                # ---------------------------------------------
                # Restrict final batch if necessary
                # ---------------------------------------------

                remaining = (
                    num_samples
                    - samples_processed
                )

                if len(image_batch) > remaining:

                    image_batch = image_batch[:remaining]

                # ---------------------------------------------
                # Create batch
                # ---------------------------------------------

                batch = torch.stack(
                    image_batch,
                    dim=0
                )

                # ---------------------------------------------
                # FP32 inference
                # ---------------------------------------------

                batch = batch.to(
                    self.device,
                    dtype=torch.float32
                )

                self.current_activations = {}

                with torch.no_grad():

                    _ = self.model(batch)

                # ---------------------------------------------
                # Store FP32 activations
                # ---------------------------------------------

                for layer_name in self.target_layer_names:

                    if layer_name not in self.current_activations:

                        raise RuntimeError(
                            f"No activation captured for "
                            f"{layer_name}"
                        )

                    activation = (
                        self.current_activations[
                            layer_name
                        ]
                    )

                    self.activation_cache[
                        layer_name
                    ].append(activation)

                # ---------------------------------------------
                # Update counter
                # ---------------------------------------------

                samples_processed += len(image_batch)

                print(
                    f"Processed "
                    f"{samples_processed}/{num_samples} "
                    f"calibration images"
                )

                # ---------------------------------------------
                # Clear batch
                # ---------------------------------------------

                image_batch = []

            # -----------------------------------------------------
            # Stop streaming
            # -----------------------------------------------------

            if samples_processed >= num_samples:
                break

        # ---------------------------------------------------------
        # Verification
        # ---------------------------------------------------------

        print("\n==============================================")
        print("CALIBRATION EXTRACTION COMPLETED")
        print("==============================================")

        print(
            f"Total images processed: "
            f"{samples_processed}"
        )

        for layer_name in self.activation_cache:

            batches = self.activation_cache[layer_name]

            print(
                f"\nLayer: {layer_name}"
            )

            print(
                f"Cached batches: {len(batches)}"
            )

            if len(batches) > 0:

                print(
                    f"Activation shape: "
                    f"{batches[0].shape}"
                )

                print(
                    f"Activation dtype: "
                    f"{batches[0].dtype}"
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
