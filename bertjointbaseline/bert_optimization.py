# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applys a warmup schedule on a given learning rate decay schedule."""

    def __init__(
            self,
            initial_learning_rate,
            decay_schedule_fn,
            warmup_steps,
            power=1.0,
            name=None):
        super(WarmUp, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or 'WarmUp') as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = (
                    self.initial_learning_rate *
                    tf.math.pow(warmup_percent_done, self.power))
            return tf.cond(global_step_float < warmup_steps_float,
                           lambda: warmup_learning_rate,
                           lambda: self.decay_schedule_fn(step),
                           name=name)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_schedule_fn': self.decay_schedule_fn,
            'warmup_steps': self.warmup_steps,
            'power': self.power,
            'name': self.name
        }


# def create_optimizer(init_lr, num_train_steps, num_warmup_steps):
#     """Creates an optimizer with learning rate schedule."""
#     # Implements linear decay of the learning rate.
#     learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
#         initial_learning_rate=init_lr,
#         decay_steps=num_train_steps,
#         end_learning_rate=0.0)
#     if num_warmup_steps:
#         learning_rate_fn = WarmUp(initial_learning_rate=init_lr,
#                                   decay_schedule_fn=learning_rate_fn,
#                                   warmup_steps=num_warmup_steps)
#     optimizer = AdamWeightDecay(
#         learning_rate=learning_rate_fn,
#         weight_decay_rate=0.01,
#         beta_1=0.9,
#         beta_2=0.999,
#         epsilon=1e-6,
#         exclude_from_weight_decay=['layer_norm', 'bias'])
#     return optimizer

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
    """Creates an optimizer training op."""
    global_step = tf.compat.v1.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.compat.v1.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.compat.v1.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


#
# class AdamWeightDecay(tf.keras.optimizers.Adam):
#     """Adam enables L2 weight decay and clip_by_global_norm on gradients.
#
#     Just adding the square of the weights to the loss function is *not* the
#     correct way of using L2 regularization/weight decay with Adam, since that will
#     interact with the m and v parameters in strange ways.
#
#     Instead we want ot decay the weights in a manner that doesn't interact with
#     the m/v parameters. This is equivalent to adding the square of the weights to
#     the loss with plain (non-momentum) SGD.
#     """
#
#     def __init__(self,
#                  learning_rate=0.001,
#                  beta_1=0.9,
#                  beta_2=0.999,
#                  epsilon=1e-7,
#                  amsgrad=False,
#                  weight_decay_rate=0.0,
#                  include_in_weight_decay=None,
#                  exclude_from_weight_decay=None,
#                  name='AdamWeightDecay',
#                  **kwargs):
#         super(AdamWeightDecay, self).__init__(
#             learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
#         self.weight_decay_rate = weight_decay_rate
#         self._include_in_weight_decay = include_in_weight_decay
#         self._exclude_from_weight_decay = exclude_from_weight_decay
#
#     @classmethod
#     def from_config(cls, config):
#         """Creates an optimizer from its config with WarmUp custom object."""
#         custom_objects = {'WarmUp': WarmUp}
#         return super(AdamWeightDecay, cls).from_config(
#             config, custom_objects=custom_objects)
#
#     def _prepare_local(self, var_device, var_dtype, apply_state):
#         super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
#                                                     apply_state)
#         apply_state['weight_decay_rate'] = tf.constant(
#             self.weight_decay_rate, name='adam_weight_decay_rate')
#
#     def _decay_weights_op(self, var, learning_rate, apply_state):
#         do_decay = self._do_use_weight_decay(var.name)
#         if do_decay:
#             return var.assign_sub(
#                 learning_rate * var *
#                 apply_state['weight_decay_rate'],
#                 use_locking=self._use_locking)
#         return tf.no_op()
#
#     def apply_gradients(self, grads_and_vars, name=None):
#         grads, tvars = list(zip(*grads_and_vars))
#         (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
#         return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars))
#
#     def _get_lr(self, var_device, var_dtype, apply_state):
#         """Retrieves the learning rate with the given state."""
#         if apply_state is None:
#             return self._decayed_lr_t[var_dtype], {}
#
#         apply_state = apply_state or {}
#         coefficients = apply_state.get((var_device, var_dtype))
#         if coefficients is None:
#             coefficients = self._fallback_apply_state(var_device, var_dtype)
#             apply_state[(var_device, var_dtype)] = coefficients
#
#         return coefficients['lr_t'], dict(apply_state=apply_state)
#
#     def _resource_apply_dense(self, grad, var, apply_state=None):
#         lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
#         decay = self._decay_weights_op(var, lr_t, apply_state)
#         with tf.control_dependencies([decay]):
#             return super(AdamWeightDecay, self)._resource_apply_dense(
#                 grad, var, **kwargs)
#
#     def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
#         lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
#         decay = self._decay_weights_op(var, lr_t, apply_state)
#         with tf.control_dependencies([decay]):
#             return super(AdamWeightDecay, self)._resource_apply_sparse(
#                 grad, var, indices, **kwargs)
#
#     def get_config(self):
#         config = super(AdamWeightDecay, self).get_config()
#         config.update({
#             'weight_decay_rate': self.weight_decay_rate,
#         })
#         return config
#
#     def _do_use_weight_decay(self, param_name):
#         """Whether to use L2 weight decay for `param_name`."""
#         if self.weight_decay_rate == 0:
#             return False
#
#         if self._include_in_weight_decay:
#             for r in self._include_in_weight_decay:
#                 if re.search(r, param_name) is not None:
#                     return True
#
#         if self._exclude_from_weight_decay:
#             for r in self._exclude_from_weight_decay:
#                 if re.search(r, param_name) is not None:
#                     return False
#         return True

class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.compat.v1.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.compat.v1.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
