# Implementation was derived from:
# https://github.com/keras-team/tf-keras/issues/301#issuecomment-2026757030 by @AIGideon
# with modification for supporting TF2.16 (and Keras 2.16)

import tf_keras as keras
import tensorflow as tf
from tf_keras.src.optimizers.utils import filter_empty_gradients

class GAOptimizer(keras.optimizers.Optimizer):
  """Optimizer wrapper for gradient accumulation."""

  def __init__(
    self,
    optimizer: keras.optimizers.Optimizer,
    accum_steps: int = 2,
    name: str = "GAOptimizer",
    **kwargs
  ):
    """Construct a new GAOptimizer optimizer.

    Adding support for sparse tensors was tricky, but this resource was
    helpful. Note that you need to implement both _resource_apply_sparse()
    and _resource_apply_sparse_duplicate_indices() for it to work as
    intended.

    See here for more information regarding implementation:
    * https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/average_wrapper.py#L93  # noqa

    Args:
      optimizer: str or `keras.optimizers.Optimizer` that will be
        used to compute and apply gradients.
      accum_steps: int > 1. Update gradient in every accumulation steps.
      name: Optional name for the operations created when applying
        gradients. Defaults to "GradientAccumulateOptimizer".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`,
        `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
        norm; `clipvalue` is clip gradients by value, `decay` is
        included for backward compatibility to allow time inverse
        decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super().__init__(name=name, **kwargs)
    self._optimizer = keras.optimizers.get(optimizer)
    self._accum_gradients = None
    self._accum_steps = accum_steps
    self._built = False

    if not accum_steps >= 2:
      raise ValueError(
        "`accum_steps` must be an integer >= 2. "
        f"Received: accum_steps={accum_steps}"
      )

  def build(self, var_list):
    if hasattr(self, "_built") and self._built:
      return
    super().build(var_list)

    self._optimizer.build(var_list)
    self._accum_gradients = []
    for var in var_list:
      self._accum_gradients.append(
        self.add_variable_from_reference(model_variable=var, variable_name='ga')
      )
    self._built = True

  def apply_gradients(self, grads_and_vars, name=None, skip_gradients_aggregation=False, **kwargs):
    # `experimental_aggregate_gradients` is an arg in `apply_gradients` of
    # v2 optimizer -- the reverse of `skip_gradients_aggregation`.
    # We read it from kwargs for backward compatibility.
    experimental_aggregate_gradients = kwargs.pop(
      "experimental_aggregate_gradients", True
    )
    run_with_dtensor = (
      # `_run_with_dtensor` is for dtensor based strategy scope, and
      # `_mesh` is when user explicitly specify the mesh setting for
      # optimizer.
      self._optimizer._run_with_dtensor
      or self._optimizer._mesh
    )
    if (
      not skip_gradients_aggregation
      and experimental_aggregate_gradients
      and not run_with_dtensor
    ):
      grads_and_vars = self._optimizer.aggregate_gradients(grads_and_vars)

    grads_and_vars = list(grads_and_vars)
    grads, trainable_variables = zip(*grads_and_vars)
    scope_name = name or self.name or "optimizer"
    with tf.name_scope(scope_name):
      with tf.init_scope():
        # Lift variable creation to init scope to avoid environment issues.
        self.build(trainable_variables)
      grads_and_vars = filter_empty_gradients(grads_and_vars)

      # iteration = self._internal_apply_gradients(grads_and_vars)
      grads, trainable_variables = zip(*grads_and_vars)

      is_update_step = (self.iterations + 1) % self._accum_steps == 0
      # `trainable_variables` might have been filtered in previous
      # processing steps, so we need to ensure the correct mapping between
      # `self._accum_gradients` and `trainable_variables`
      # acc_grads = self._accum_gradients
      acc_grads = [
        self._accum_gradients[self._index_dict[self._var_key(v)]]
        for v in trainable_variables
      ]

      def _cross_replica_apply_gradients(strategy, the_acc_grads, the_grads, the_trainable_variables):
        def _update_step_fn():
          strategy.extended.call_for_each_replica(
            self._apply_accumulated_gradients,
            args=(the_acc_grads, the_grads, the_trainable_variables))

        def _accumulate_step_fn():
          strategy.extended.call_for_each_replica(
            self._accumulate_gradients,
            args=(the_acc_grads, the_grads))

        tf.cond(
          is_update_step,
          _update_step_fn,
          _accumulate_step_fn,
        )

      tf.distribute.get_replica_context().merge_call(
        _cross_replica_apply_gradients, args=(acc_grads, grads, trainable_variables))

      return self.iterations.assign_add(1)

  def _apply_accumulated_gradients(self, acc_grads, grads, trainable_variables):
    """Apply accumulated gradients by inner optimizer.
    """
    # Accumulate grads and normalize by accum_steps
    steps = self._accum_steps
    accnorm_grads = [
      (acc_g + g) / steps for acc_g, g in zip(acc_grads, grads)
    ]
    # skip_gradients_aggregation=True because this optimizer already did aggregate_gradients() in the begining of apply_gradients()
    self._optimizer.apply_gradients(zip(accnorm_grads, trainable_variables), skip_gradients_aggregation=True)
    # Reset gradient accumulators
    self._reset_accumulated_gradients()

  @staticmethod
  def _accumulate_gradients(acc_grads, grads):
    for acc_g, g in zip(acc_grads, grads):
      acc_g.assign_add(g)

  def _reset_accumulated_gradients(self):
    for g_acc in self._accum_gradients:
      g_acc.assign(tf.zeros(g_acc.shape, dtype=g_acc.dtype))

  def update_step(self, gradient, variable):
    # Not used because inner optimizer is used for update_step
    pass

  @property
  def gradients(self):
    """The accumulated gradients on the current replica.

    Returns:
      Current accumulated gradients in optimizer.
    """
    if not self._accum_gradients:
      raise ValueError(
        "The accumulator should be called first to initialize the"
        "gradients"
      )
    return list(
      gradient.read_value() if gradient is not None else gradient
      for gradient in self._accum_gradients
    )

  @property
  def inner_optimizer(self):
    """Returns the wrapped optimizer which this GAOptimizer is wrapping."""
    return self._optimizer

  @property
  def inner_iterations(self):
    # iterations (of GAOptimizer) has different (larger) value from iterations of inner optimizer
    # iterations will be in range of [accum_steps * inner_iterations, accum_steps * (inner_iterations+1))
    return self._optimizer.iterations

  @inner_iterations.setter
  def inner_iterations(self, variable):
    # iterations (of GAOptimizer) has different (larger) value from iterations of inner optimizer
    # iterations will be in range of [accum_steps * inner_iterations, accum_steps * (inner_iterations+1))
    self._optimizer.iterations = variable

  @property
  def learning_rate(self):
    """Returns the learning rate of the wrapped optimizer."""
    # GAOptimizer itself doesn't have valid learning rate, so borrows from inner optimizer
    return self._optimizer.learning_rate

  @learning_rate.setter
  def learning_rate(self, learning_rate):
    """Sets the learning rate of the wrapped optimizer.

    Args:
      learning_rate: which learning rate to set in the wrapped optimizer.
    """
    # GAOptimizer itself doesn't have valid learning rate, so borrows from inner optimizer
    self._optimizer.learning_rate = learning_rate

  @property
  def lr(self):
    return self._optimizer.learning_rate

  @lr.setter
  def lr(self, learning_rate):
    self._optimizer.learning_rate = learning_rate

  def get_config(self):
    config = super().get_config()
    config.update({
      "optimizer": keras.optimizers.serialize(self._optimizer),
      "accum_steps": self._accum_steps,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Gets config of original optimizer and deserializes it."""
    optimizer = keras.optimizers.deserialize(
      config.pop("optimizer"), custom_objects=custom_objects
    )
    return cls(optimizer, **config)

keras.utils.get_custom_objects().update({'GAOptimizer': GAOptimizer})
