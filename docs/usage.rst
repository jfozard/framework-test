Usage
=====

Here is a basic example:

.. math::
   :label: euler-step

   y_{k+1} = y_k + \Delta t\, f(t_k, y_k)

.. code-block:: python

   import jax.numpy as jnp
   from framework_test import forward_euler_solver

   def ode(t, y):
       return -y

   times = jnp.array([0.0, 1.0])
   initial_value = jnp.array([1.0])

   sol = forward_euler_solver(ode, times, initial_value)
