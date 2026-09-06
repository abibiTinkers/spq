"""
SPQ - Module 4
Optimization Module

Purpose:
    Optimize the quantization scale so that the similarity matrix
    of quantized/dequantized activations remains close to the FP32
    similarity matrix.

Objective:
    L = (1 / B^2) * sum_ij (G_ij - G_hat_ij)^2

Update:
    S* = S - eta * dL/dS

Note:
    Hard floor() + INT8 conversion is not differentiable.
    Therefore, a Straight-Through Estimator (STE) is used during
    optimization. The final output is converted to real INT8.
"""

import torch
import torch.nn.functional as F


class SPQOptimizationModule:

    def __init__(
        self,
        learning_rate=1e-3,
        epochs=20,
        step_size=5,
        gamma=0.5,
        q_min=-128,
        q_max=127,
        eps=1e-8
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.step_size = step_size
        self.gamma = gamma
        self.q_min = q_min
        self.q_max = q_max
        self.eps = eps

    def similarity_matrix(self, features):
        """
        Calculate:

            A(i,:) = X(i,:) / ||X(i,:)||2
            G = A @ A^T
        """

        x = features.reshape(features.shape[0], -1)

        x = F.normalize(
            x,
            p=2,
            dim=1,
            eps=self.eps
        )

        return x @ x.T

    def fake_quantize_ste(self, x, scale, zero_point):
        """
        Differentiable fake quantization using STE.

        Forward:
            q = clip(floor(x/S) + Z)
            x_hat = S(q-Z)

        Backward:
            Gradient is passed through the quantization operation.
        """

        scale = torch.clamp(
            scale,
            min=self.eps
        )

        q = torch.floor(x / scale) + zero_point

        q = torch.clamp(
            q,
            self.q_min,
            self.q_max
        )

        dequantized = scale * (
            q - zero_point
        )

        # Straight-Through Estimator
        return x + (
            dequantized - x
        ).detach()

    def calculate_initial_scale(self, x):
        """
        Calculate initial symmetric INT8 scale.
        """

        max_abs = torch.max(
            torch.abs(x)
        ).detach()

        scale = max_abs / 127.0

        if scale.item() <= 0:
            scale = torch.tensor(
                1.0,
                device=x.device
            )

        return scale.item()

    def optimize_batch(
        self,
        fp32_features,
        initial_scale=None,
        zero_point=0.0
    ):
        """
        Optimize quantization scale for one activation batch.
        """

        x = fp32_features.detach().float()

        if initial_scale is None:
            initial_scale = self.calculate_initial_scale(x)

        scale = torch.nn.Parameter(
            torch.tensor(
                max(initial_scale, self.eps),
                dtype=torch.float32,
                device=x.device
            )
        )

        zero_point_tensor = torch.tensor(
            zero_point,
            dtype=torch.float32,
            device=x.device
        )

        # ------------------------------------------
        # FP32 target similarity matrix
        # ------------------------------------------

        fp32_similarity = self.similarity_matrix(
            x
        ).detach()

        # ------------------------------------------
        # Adam optimizer
        # ------------------------------------------

        optimizer = torch.optim.Adam(
            [scale],
            lr=self.learning_rate
        )

        # ------------------------------------------
        # Step-decay scheduler
        # ------------------------------------------

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma
        )

        loss_history = []

        # ------------------------------------------
        # Optimization loop
        # ------------------------------------------

        for epoch in range(self.epochs):

            optimizer.zero_grad()

            # Differentiable fake quantization
            fake_quantized = self.fake_quantize_ste(
                x,
                scale,
                zero_point_tensor
            )

            # Quantized similarity matrix
            quantized_similarity = self.similarity_matrix(
                fake_quantized
            )

            # --------------------------------------
            # MSE objective
            # --------------------------------------

            loss = torch.mean(
                (
                    fp32_similarity
                    - quantized_similarity
                ) ** 2
            )

            # Backpropagation
            loss.backward()

            # Adam update
            optimizer.step()

            # Scale must remain positive
            with torch.no_grad():
                scale.clamp_(
                    min=self.eps
                )

            # StepLR update
            scheduler.step()

            loss_history.append(
                loss.item()
            )

        # ------------------------------------------
        # Optimized scale
        # ------------------------------------------

        optimized_scale = scale.detach().item()

        # ------------------------------------------
        # REAL INT8 quantization
        # ------------------------------------------

        q = torch.floor(
            x / optimized_scale
        ) + zero_point_tensor

        q = torch.clamp(
            q,
            self.q_min,
            self.q_max
        )

        int8_tensor = q.to(torch.int8)

        # ------------------------------------------
        # Dequantization
        # ------------------------------------------

        dequantized_tensor = (
            optimized_scale
            * (
                int8_tensor.float()
                - zero_point_tensor
            )
        )

        # ------------------------------------------
        # Final similarity matrix
        # ------------------------------------------

        final_similarity = self.similarity_matrix(
            dequantized_tensor
        ).detach()

        # ------------------------------------------
        # Final errors
        # ------------------------------------------

        final_mse = torch.mean(
            (
                fp32_similarity
                - final_similarity
            ) ** 2
        ).item()

        final_mae = torch.mean(
            torch.abs(
                fp32_similarity
                - final_similarity
            )
        ).item()

        return {
            "initial_scale": initial_scale,
            "optimized_scale": optimized_scale,
            "zero_point": zero_point,
            "loss_history": loss_history,
            "fp32_similarity": fp32_similarity.cpu(),
            "optimized_similarity": final_similarity.cpu(),
            "int8_tensor": int8_tensor.cpu(),
            "dequantized_tensor": dequantized_tensor.cpu(),
            "final_mse": final_mse,
            "final_mae": final_mae
        }


def run_module4(
    fp32_features,
    learning_rate=1e-3,
    epochs=20,
    step_size=5,
    gamma=0.5
):
    """
    Simple function for Colab/notebook use.
    """

    optimizer = SPQOptimizationModule(
        learning_rate=learning_rate,
        epochs=epochs,
        step_size=step_size,
        gamma=gamma
    )

    return optimizer.optimize_batch(
        fp32_features
    )


# --------------------------------------------------
# Self-test
# --------------------------------------------------

if __name__ == "__main__":

    torch.manual_seed(42)

    # Example activation:
    # [Batch, Channels, Height, Width]
    sample = torch.randn(
        32,
        16,
        7,
        7
    )

    result = run_module4(
        sample,
        learning_rate=1e-3,
        epochs=5,
        step_size=3,
        gamma=0.5
    )

    print("SPQ Module 4 Self-Test")
    print(
        "Initial scale:",
        result["initial_scale"]
    )
    print(
        "Optimized scale:",
        result["optimized_scale"]
    )
    print(
        "Final MSE:",
        result["final_mse"]
    )
    print(
        "Final MAE:",
        result["final_mae"]
    )
    print(
        "INT8 dtype:",
        result["int8_tensor"].dtype
    )
