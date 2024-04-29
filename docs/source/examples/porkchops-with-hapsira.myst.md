---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Porkchops with hapsira

Porkchops are also known as mission design curves since they show different parameters used to design the ballistic trajectories for the targetting problem such as:

* Time of flight (TFL)
* Launch energy (C3L)
* Arrival velocity (VHP)

For the moment, hapsira is only capable of creating these mission plots between `hapsira.bodies` objects. However, it is intended for future versions to make it able for plotting porkchops between NEOs also.

+++

## Basic modules
For creating a porkchop plot with hapsira, we need to import the `porkchop` function from the `hapsira.plotting.porkchop` module. Also, two `hapsira.bodies` are necessary for computing the targetting problem associated. Finally by making use of `time_range`, a very useful function available at `hapsira.utils` it is possible to define a span of launching and arrival dates for the problem:

```{code-cell}
from astropy import units as u

from hapsira.bodies import Earth, Mars
from hapsira.plotting.porkchop import PorkchopPlotter
from hapsira.util import time_range

launch_span = time_range("2005-04-30", end="2005-10-07")
arrival_span = time_range("2005-11-16", end="2006-12-21")
```

## Plot that porkchop!

All that we must do is pass the two bodies, the two time spans and some extra plotting parameters realted to different information along the figure such us:

* If we want hapsira to plot time of flight lines: `tfl=True/False`
* If we want hapsira to plot arrival velocity: `vhp=True/False`
* The maximum value for C3 to be ploted: `max_c3=45 * u.km**2 / u.s**2` (by default)

```{code-cell}
:tags: [nbsphinx-thumbnail]

porkchop_plot = PorkchopPlotter(Earth, Mars, launch_span, arrival_span)
dv_dpt, dv_arr, c3dpt, c3arr, tof = porkchop_plot.porkchop()
```

## NASA's same porkchop

We can compare previous porkchop with the ones made by NASA for those years.

![Porkchop to Mars](porkchop_mars.png)
