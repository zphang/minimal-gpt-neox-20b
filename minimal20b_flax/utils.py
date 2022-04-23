import numpy as np
import jax.interpreters.pxla
import jax.numpy as jnp


def replicate_to_devices(array, devices=None):
    if devices is None:
        devices = jax.local_devices()
    num_devices = len(devices)
    num_dims = array.ndim
    device_buffers = [
        jax.device_put(array, device)
        for device in devices
    ]
    sharding = tuple([jax.interpreters.pxla.NoSharding() for _ in range(num_dims)])
    # Assuming mesh=("dp", "tp")
    mesh_mapping = (
        jax.interpreters.pxla.Replicated(1),
        jax.interpreters.pxla.Replicated(num_devices),
    )
    sharding_spec = jax.interpreters.pxla.ShardingSpec(sharding, mesh_mapping)
    return jax.interpreters.pxla.make_sharded_device_array(
        aval=jax.ShapedArray(array.shape, jnp.float16),
        sharding_spec=sharding_spec,
        device_buffers=device_buffers,
    )


def shard_to_devices(array, axis, devices=None):
    if devices is None:
        devices = jax.local_devices()
    num_devices = len(devices)
    num_dims = array.ndim
    split_arrays = np.split(array, num_devices, axis=axis)
    device_buffers = [
        jax.device_put(split_array, device)
        for split_array, device
        in zip(split_arrays, devices)
    ]
    sharding = [jax.interpreters.pxla.NoSharding() for _ in range(num_dims)]
    sharding[axis] = jax.interpreters.pxla.Chunked((num_devices,))
    sharding = tuple(sharding)
    # Assuming mesh=("dp", "tp")
    mesh_mapping = (
        jax.interpreters.pxla.Replicated(1),
        jax.interpreters.pxla.ShardedAxis(0),
    )
    sharding_spec = jax.interpreters.pxla.ShardingSpec(sharding, mesh_mapping)
    return jax.interpreters.pxla.make_sharded_device_array(
        aval=jax.ShapedArray(array.shape, jnp.float16),
        sharding_spec=sharding_spec,
        device_buffers=device_buffers,
    )


def split_to_device_buffers(array, axis, devices=None):
    if devices is None:
        devices = jax.local_devices()
    num_devices = len(devices)
    split_arrays = np.split(array, num_devices, axis=axis)
    device_buffers = [
        jax.device_put(split_array, device)
        for split_array, device
        in zip(split_arrays, devices)
    ]
    return device_buffers


def wrap_device_buffers_in_sharded_device_array(device_buffers, array_shape, axis, devices=None):
    if devices is None:
        devices = jax.local_devices()
    num_devices = len(devices)
    num_dims = len(array_shape)
    sharding = [jax.interpreters.pxla.NoSharding() for _ in range(num_dims)]
    sharding[axis] = jax.interpreters.pxla.Chunked((num_devices,))
    sharding = tuple(sharding)
    mesh_mapping = (
        jax.interpreters.pxla.Replicated(1),
        jax.interpreters.pxla.ShardedAxis(0),
    )
    sharding_spec = jax.interpreters.pxla.ShardingSpec(sharding, mesh_mapping)
    return jax.interpreters.pxla.make_sharded_device_array(
        aval=jax.ShapedArray(array_shape, jnp.float16),
        sharding_spec=sharding_spec,
        device_buffers=device_buffers,
    )
