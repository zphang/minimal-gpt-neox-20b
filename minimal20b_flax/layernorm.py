from typing import Any, Callable, Iterable, Tuple, Union

import flax.linen as nn
from jax.nn import initializers
import jax.numpy as jnp
import jax.lax as lax

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any    # this could be a real type?

Axes = Union[int, Iterable[int]]


class LayerNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

    @nn.compact
    def __call__(self, x):
        """Applies layer normalization on the input.

        Args:
            x: the inputs

        Returns:
            Normalized inputs (the same shape as inputs).
        """
        reduction_axes = (-1,)
        feature_axes = (-1,)

        # TODO(jheek) suport axis_name for model parallelism?
        mean, var = _compute_stats(x, reduction_axes)

        return _normalize(
                self, x, mean, var, reduction_axes, feature_axes,
                self.dtype, self.param_dtype, self.epsilon,
                self.use_bias, self.use_scale,
                self.bias_init, self.scale_init)


def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, Iterable):
        axes = (axes,)
    return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _abs_sq(x):
    """Computes the elementwise square of the absolute value |x|^2."""
    if jnp.iscomplexobj(x):
        return lax.square(lax.real(x)) + lax.square(lax.imag(x))
    else:
        return lax.square(x)


def _compute_stats(x: Array, axes: Axes):
    """Computes mean and variance statistics.

    This implementation takes care of a few important details:
    - Computes in float32 precision for half precision inputs
    -    mean and variance is computable in a single XLA fusion,
        by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
    - Clips negative variances to zero which can happen due to
        roundoff errors. This avoids downstream NaNs.
    - Supports averaging across a parallel axis and subgroups of a parallel axis
        with a single `lax.pmean` call to avoid latency.

    Arguments:
        x: Input array.
        axes: The axes in ``x`` to compute mean and variance statistics for.

    Returns:
        A pair ``(mean, var)``.
    """
    # promote x to at least float32, this avoids half precision computation
    # but preserves double or complex floating points
    x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
    mean = jnp.mean(x, axes)
    diff = x - mean[..., None]
    var = jnp.mean(_abs_sq(diff), axes)
    return mean, var


def _normalize(mdl: nn.Module, x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Dtype, param_dtype: Dtype,
               epsilon: float,
               use_bias: bool, use_scale: bool,
               bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
               scale_init: Callable[[PRNGKey, Shape, Dtype], Array]):
    """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

    Arguments:
        mdl: Module to apply the normalization in (normalization params will reside
            in this module).
        x: The input.
        mean: Mean to use for normalization.
        var: Variance to use for normalization.
        reduction_axes: The axes in ``x`` to reduce.
        feature_axes: Axes containing features. A separate bias and scale is learned
            for each specified feature.
        dtype: Dtype of the returned result.
        param_dtype: Dtype of the parameters.
        epsilon: Normalization epsilon.
        use_bias: If true, add a bias term to the output.
        use_scale: If true, scale the output.
        bias_init: Initialization function for the bias term.
        scale_init: Initialization function for the scaling function.

    Returns:
        The normalized input.
    """
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])
    y = x - mean
    mul = lax.rsqrt(var + epsilon)
    if use_scale:
        scale = mdl.param('scale', scale_init, reduced_feature_shape,
                          param_dtype).reshape(feature_shape)
        mul *= scale
    y *= mul
    if use_bias:
        bias = mdl.param('bias', bias_init, reduced_feature_shape,
                         param_dtype).reshape(feature_shape)
        y += bias
    return jnp.asarray(y, dtype)
