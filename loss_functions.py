import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, List, Dict

# Define constants and configuration
VELOCITY_THRESHOLD = 0.5
FLOW_THEORY_THRESHOLD = 0.2
LOSS_FUNCTION_CONFIG = {
    'velocity_threshold': VELOCITY_THRESHOLD,
    'flow_theory_threshold': FLOW_THEORY_THRESHOLD
}

# Define exception classes
class LossFunctionError(Exception):
    """Base class for loss function exceptions."""
    pass

class InvalidLossFunctionError(LossFunctionError):
    """Raised when an invalid loss function is specified."""
    pass

class InvalidInputError(LossFunctionError):
    """Raised when invalid input is provided to a loss function."""
    pass

# Define a logger
logger = logging.getLogger(__name__)

class LossFunction(nn.Module):
    """
    Base class for custom loss functions.

    Attributes:
        name (str): The name of the loss function.
        config (Dict): The configuration for the loss function.
    """
    def __init__(self, name: str, config: Dict):
        super(LossFunction, self).__init__()
        self.name = name
        self.config = config

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
        raise NotImplementedError

class VelocityThresholdLoss(LossFunction):
    """
    Loss function based on the velocity threshold.

    Attributes:
        threshold (float): The velocity threshold.
    """
    def __init__(self, config: Dict):
        super(VelocityThresholdLoss, self).__init__('velocity_threshold', config)
        self.threshold = config['velocity_threshold']

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss based on the velocity threshold.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
        velocity = torch.abs(input - target)
        loss = torch.mean(torch.where(velocity > self.threshold, velocity, torch.zeros_like(velocity)))
        return loss

class FlowTheoryLoss(LossFunction):
    """
    Loss function based on the flow theory.

    Attributes:
        threshold (float): The flow theory threshold.
    """
    def __init__(self, config: Dict):
        super(FlowTheoryLoss, self).__init__('flow_theory', config)
        self.threshold = config['flow_theory_threshold']

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss based on the flow theory.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
        flow = torch.abs(input - target)
        loss = torch.mean(torch.where(flow > self.threshold, flow, torch.zeros_like(flow)))
        return loss

class CompositeLoss(LossFunction):
    """
    Composite loss function that combines multiple loss functions.

    Attributes:
        loss_functions (List[LossFunction]): The list of loss functions to combine.
    """
    def __init__(self, config: Dict, loss_functions: List[LossFunction]):
        super(CompositeLoss, self).__init__('composite', config)
        self.loss_functions = loss_functions

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the composite loss.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed composite loss.
        """
        losses = [loss_function(input, target) for loss_function in self.loss_functions]
        return torch.mean(torch.stack(losses))

def get_loss_function(config: Dict, name: str) -> LossFunction:
    """
    Get a loss function based on the name.

    Args:
        config (Dict): The configuration for the loss function.
        name (str): The name of the loss function.

    Returns:
        LossFunction: The loss function instance.

    Raises:
        InvalidLossFunctionError: If the loss function name is invalid.
    """
    if name == 'velocity_threshold':
        return VelocityThresholdLoss(config)
    elif name == 'flow_theory':
        return FlowTheoryLoss(config)
    elif name == 'composite':
        # Create a composite loss function with multiple loss functions
        loss_functions = [VelocityThresholdLoss(config), FlowTheoryLoss(config)]
        return CompositeLoss(config, loss_functions)
    else:
        raise InvalidLossFunctionError(f"Invalid loss function name: {name}")

def validate_input(input: torch.Tensor, target: torch.Tensor) -> None:
    """
    Validate the input and target tensors.

    Args:
        input (torch.Tensor): The input tensor.
        target (torch.Tensor): The target tensor.

    Raises:
        InvalidInputError: If the input or target is invalid.
    """
    if not isinstance(input, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise InvalidInputError("Input and target must be tensors")
    if input.shape != target.shape:
        raise InvalidInputError("Input and target must have the same shape")

def compute_loss(loss_function: LossFunction, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss using the given loss function.

    Args:
        loss_function (LossFunction): The loss function instance.
        input (torch.Tensor): The input tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The computed loss.
    """
    validate_input(input, target)
    return loss_function(input, target)

# Example usage
if __name__ == "__main__":
    # Create a loss function instance
    config = LOSS_FUNCTION_CONFIG
    loss_function = get_loss_function(config, 'velocity_threshold')

    # Create input and target tensors
    input_tensor = torch.randn(10, 10)
    target_tensor = torch.randn(10, 10)

    # Compute the loss
    loss = compute_loss(loss_function, input_tensor, target_tensor)
    logger.info(f"Computed loss: {loss.item()}")