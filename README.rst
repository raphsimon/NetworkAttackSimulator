Network Attack Simulator
========================

|docs|

Network Attack Simulator (NASim) is a simulated computer network complete with vulnerabilities, scans and exploits designed to be used as a testing environment for AI agents and planning techniques applied to network penetration testing.


StochNASim extension
----------------------
StochNASim is an extension of NASim that adds stochasticity to the environment, allowing for more realistic simulations. Every episode, a new environment is generated according to the provided parameters. Addtionally, it supports variable sized networks, allow for the learning of more robust policies.

Example usage:

.. code-block:: python

    env = gym.make('StochPO-v0',
     min_num_hosts=5,
     max_num_hosts=8,
     exploit_probs=0.9,
     privesc_probs=0.9,
     seed=2,
     render_mode='human')
                     

Installation
------------

Installing the environment can be done via pip::

  $ pip install .

