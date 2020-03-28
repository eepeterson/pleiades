=================
The Pleiades Code
=================

The `Pleiades code <https://www.github.com/eepeterson/pleiades>`_ is designed
for computing axisymmetric magnetic fields from combinations of electromagnets
and permanent magnets with a focus on modeling plasma physics experiments. The
project originated at the `Wisconsin Plasma Physics Laboratory
<https://wippl.wisc.edu>`_ during experiments on the novel confinement device
known as the `Big Red Ball <https://wippl.wisc.edu/big-red-ball-brb>`_. Pleiades
is written entirely in Python for now and is intended to be an easily accessible
toolkit for computing and visualizing magnetic confinement geometries and data
simultaneously. 

Aside from simple computations of magnetic fields, there is a plasma MHD
equilibrium solver incorporated into the package as well, which has been used to
generate magnetic mirror equilibria and input files suitable for being run with
the `GENRAY <https://www.compxco.com/genray.html>`_ and `CQL3D
<https://www.compxco.com/cql3d.html>`_ codes for the design of the new Wisconsin
HTS Axisymmetric Mirror (WHAM) and its eventual upgrade to a neutron source
(WHAM-NS).

The Pleiades code is available on `github
<https://www.github.com/eepeterson/pleiades>`_. Any and all feedback and
contributions are welcome!


.. only:: html

   --------
   Contents
   --------

.. toctree::
    :maxdepth: 1

    quickinstall
    examples/index
    pythonapi/index
    gallery
    publications
    resources
    license
