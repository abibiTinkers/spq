"""
SPQ - Module 3
Similarity Matrix Calculation Module

Purpose:
    1. Take FP32 and quantized/dequantized feature maps.
    2. Flatten the feature maps.
    3. Apply L2 normalization to each feature vector.
    4. Calculate pairwise similarity matrices.
    5. Compare FP32 and quantized representations.

Mathematical formulation:

    A_(i,:) = X_(i,:) / ||X_(i,:)||_2

    G = A . A^T

Where:
    X = feature representation
    A = L2-normalized feature representation
    G = pairwise similarity matrix

The output of this module is used by Module 4
for similarity-preserving optimization.
"""

import torch
import torch.nn.functional as F


class SimilarityMatrixCalculator:

    def __init__(self, eps=1e-8):
        """
        Initialize the similarity matrix calculator.

        Args:
            eps: Small value used to avoid division by zero.
        """

        self.eps = eps

        print("==============================================")
        print("SPQ MODULE 3")
        print("Similarity Matrix Calculation")
        print("==============================================")
        print("L2 normalization enabled.")
        print("Pairwise similarity calculation enabled.")

    def flatten_feature_maps(self, feature_maps):
        """
        Flatten multidimensional feature maps.

        Input:
            [B, C, H, W]

        Output:
            [B, C*H*W]

        Args:
            feature_maps: PyTorch tensor.
        """

        if not isinstance(feature_maps, torch.Tensor):
            raise TypeError(
                "feature_maps must be a torch.Tensor"
            )

        if feature_maps.ndim < 2:
            raise ValueError(
                "Feature maps must have at least 2 dimensions."
            )

        return feature_maps.reshape(
            feature_maps.shape[0],
            -1
        )

    def l2_normalize(self, features):
        """
        Apply row-wise L2 normalization.

        Formula:

            A_i = X_i / ||X_i||_2

        Args:
            features: [B, N]

        Returns:
            normalized_features: [B, N]
        """

        if features.ndim != 2:
            raise ValueError(
                "Input to l2_normalize must be 2D."
            )

        norms = torch.linalg.norm(
            features,
            p=2,
            dim=1,
            keepdim=True
        )

        norms = torch.clamp(
            norms,
            min=self.eps
        )

        normalized_features = features / norms

        return normalized_features

    def calculate_similarity_matrix(
        self,
        feature_maps
    ):
        """
        Calculate pairwise similarity matrix.

        Steps:

            1. Flatten feature maps.
            2. L2 normalize each feature vector.
            3. Calculate A @ A^T.

        Formula:

            G = A . A^T

        Args:
            feature_maps: [B, C, H, W]

        Returns:
            similarity_matrix: [B, B]
        """

        flattened = self.flatten_feature_maps(
            feature_maps
        )

        normalized = self.l2_normalize(
            flattened
        )

        similarity_matrix = torch.matmul(
            normalized,
            normalized.transpose(0, 1)
        )

        return similarity_matrix

    def calculate_fp32_similarity(
        self,
        fp32_features
    ):
        """
        Calculate the similarity matrix from
        original FP32 activation feature maps.
        """

        fp32_features = fp32_features.float()

        return self.calculate_similarity_matrix(
            fp32_features
        )

    def calculate_quantized_similarity(
        self,
        quantized_features
    ):
        """
        Calculate the similarity matrix from
        quantized/dequantized feature maps.

        The input is converted to FP32 for the
        mathematical similarity calculation.

        Note:
            The quantization itself occurs in Module 2.
        """

        quantized_features = quantized_features.float()

        return self.calculate_similarity_matrix(
            quantized_features
        )

    def compare_similarity_matrices(
        self,
        fp32_similarity,
        quantized_similarity
    ):
        """
        Compare FP32 and quantized similarity matrices.

        Calculates:

            MSE = mean((G - G_hat)^2)

        This value is used by Module 4.
        """

        if fp32_similarity.shape != quantized_similarity.shape:
            raise ValueError(
                "FP32 and quantized similarity matrices "
                "must have the same shape."
            )

        difference = (
            fp32_similarity -
            quantized_similarity
        )

        mse = torch.mean(
            difference ** 2
        )

        mean_absolute_error = torch.mean(
            torch.abs(difference)
        )

        max_absolute_error = torch.max(
            torch.abs(difference)
        )

        return {
            "mse": mse,
            "mae": mean_absolute_error,
            "max_error": max_absolute_error
        }

    def process(
        self,
        fp32_features,
        quantized_features
    ):
        """
        Complete Module 3 processing.

        Args:
            fp32_features:
                Original FP32 activation feature maps.

            quantized_features:
                Quantized/dequantized activation feature maps.

        Returns:
            Dictionary containing:
                FP32 similarity matrix
                Quantized similarity matrix
                normalized features
                comparison metrics
        """

        print("\n==============================================")
        print("STARTING MODULE 3")
        print("==============================================")

        print("\nInput FP32 feature shape:")
        print(fp32_features.shape)

        print("\nInput quantized feature shape:")
        print(quantized_features.shape)

        # --------------------------------------------------
        # FP32 processing
        # --------------------------------------------------

        fp32_flattened = self.flatten_feature_maps(
            fp32_features
        )

        fp32_normalized = self.l2_normalize(
            fp32_flattened
        )

        fp32_similarity = torch.matmul(
            fp32_normalized,
            fp32_normalized.transpose(0, 1)
        )

        # --------------------------------------------------
        # Quantized processing
        # --------------------------------------------------

        quantized_flattened = self.flatten_feature_maps(
            quantized_features.float()
        )

        quantized_normalized = self.l2_normalize(
            quantized_flattened
        )

        quantized_similarity = torch.matmul(
            quantized_normalized,
            quantized_normalized.transpose(0, 1)
        )

        # --------------------------------------------------
        # Compare similarity matrices
        # --------------------------------------------------

        comparison = self.compare_similarity_matrices(
            fp32_similarity,
            quantized_similarity
        )

        print("\n==============================================")
        print("MODULE 3 RESULTS")
        print("==============================================")

        print(
            "FP32 flattened shape      :",
            fp32_flattened.shape
        )

        print(
            "Quantized flattened shape :",
            quantized_flattened.shape
        )

        print(
            "FP32 similarity shape     :",
            fp32_similarity.shape
        )

        print(
            "Quantized similarity shape:",
            quantized_similarity.shape
        )

        print(
            "Similarity MSE             :",
            f"{comparison['mse'].item():.8f}"
        )

        print(
            "Similarity MAE             :",
            f"{comparison['mae'].item():.8f}"
        )

        print(
            "Maximum similarity error  :",
            f"{comparison['max_error'].item():.8f}"
        )

        print("\nModule 3 completed.")

        return {
            "fp32_flattened": fp32_flattened,
            "fp32_normalized": fp32_normalized,
            "fp32_similarity": fp32_similarity,

            "quantized_flattened": quantized_flattened,
            "quantized_normalized": quantized_normalized,
            "quantized_similarity": quantized_similarity,

            "mse": comparison["mse"],
            "mae": comparison["mae"],
            "max_error": comparison["max_error"]
        }


def calculate_similarity_matrix(
    feature_maps,
    eps=1e-8
):
    """
    Convenience function for calculating
    a single similarity matrix.
    """

    calculator = SimilarityMatrixCalculator(
        eps=eps
    )

    return calculator.calculate_similarity_matrix(
        feature_maps
    )
