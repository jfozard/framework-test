from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from beartype import beartype
from jaxtyping import ArrayLike, Float, jaxtyped

# import jax
# import jax.numpy as jnp

# ruff: noqa: F722, F821, UP037


@runtime_checkable
class ODEFunction(Protocol):
    """
    RHS of an ODE

    Represents the vector field ``dy/dt = f(t,y)``

    """

    def __call__(
        self, t: Float[ArrayLike, ""], y: Float[ArrayLike, "state"], /
    ) -> Float[ArrayLike, "state"]:
        """
        Evaluate the derivative ``f(t,y)``

        Args:
            t: Scalar time at which to evaluate the field
            y: Current state vector with shape ``(state,)``

        Returns:
            Derivative of y with shape ``(state,)``
        """
        ...


@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class SolutionOutput:
    """
    Class storing the solution of the ODE
    """

    times: Float[ArrayLike, "timesteps"]
    states: Float[ArrayLike, "timesteps state"]


@jaxtyped(typechecker=beartype)
def forward_euler_solver(
    fun: ODEFunction,
    times: Float[ArrayLike, "timesteps"],
    initial_value: Float[ArrayLike, "state"],
) -> SolutionOutput:
    """
    Solve an IVP using the ForwardEuler method

    Args:
        fun: Vector field ``f(t,y)``
        times: Times to evaluate the solution. Must be at least length 2 and strictly increasing.
        initial_value: Value of ``y`` at ``times[0]``

    Returns:
        Solution containing the evaluation times (shape ``(timesteps,)`` and solution states (of shape ``(timesteps, state)``)

    Raises:
        ValueError: If times is not of at least length 2 and strictly increasing

    """

    return SolutionOutput(times, np.stack([initial_value] * times.shape[0], axis=0))
