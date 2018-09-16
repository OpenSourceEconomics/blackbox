TODO
====

* We need to add the ability to restart the algorithm after its initial exploration of the function at least.

* We need to add the ability for a gentle termination of the process.

* We need to polish the codes overall.

* We need to tackle the performance bottleneck in fit_approx_model().

* We can then remove the PYTHON implementations and the legacy flag once our implementation is tested in the field.

* We need to set up a proper documentation with sound references, at least document the interface to the search() function.

* We should consider moving towards a queue model for the case where the time for the evaluation of the criterion function differs or the slave processes do not get the same performance.
