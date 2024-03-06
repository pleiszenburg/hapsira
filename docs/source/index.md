# hapsira - Astrodynamics in Python

```{image} _static/logo_text.png
---
width: 675px
align: center
---
```
**hapsira** is an open source ([MIT](https://opensource.org/licenses/MIT)) pure Python library for interactive Astrodynamics and Orbital Mechanics, with a focus on ease of use, speed, and quick visualization. It provides a simple and intuitive {ref}`API <api-reference>`, and handles physical quantities with units. It's a fork of [poliastro](http://github.com/poliastro/poliastro), aiming at continuing its development.

View the [source code](https://github.com/pleiszenburg/hapsira) of `hapsira`!

Some of its awesome features are:

- Analytical and numerical orbit propagation
- Conversion between position and velocity vectors and classical orbital elements
- Coordinate frame transformations
- Hohmann and bielliptic maneuvers computation
- Trajectory plotting
- Initial orbit determination (Lambert problem)
- Planetary ephemerides (using [SPICE kernels](https://naif.jpl.nasa.gov/naif/data.html) via [Astropy](https://www.astropy.org/))
- Computation of Near-Earth Objects (NEOs)

And more to come!

`hapsira` is developed by an open, international community. Release announcements and general discussion take place on our [mailing list](https://groups.io/g/hapsira-dev) and [chat](https://matrix.to/#/#hapsira:matrix.org).

```{eval-rst}
.. raw:: html

    <div class="classictemplate template" style="display: block;">
        <style type="text/css">
            #groupsio_embed_signup input {border:1px solid #999; -webkit-appearance:none;}
            #groupsio_embed_signup label {display:block; font-size:16px; padding-bottom:10px; font-weight:bold;}
            #groupsio_embed_signup .email {display:block; padding:8px 0; margin:0 4% 10px 0; text-indent:5px; width:58%; min-width:130px;}
            #groupsio_embed_signup {
            background:#fff; clear:left; font:14px Helvetica,Arial,sans-serif;
            }
            #groupsio_embed_signup .button {

            width:25%; margin:0 0 10px 0; min-width:90px;
            background-image: linear-gradient(to bottom,#337ab7 0,#265a88 100%);
            background-repeat: repeat-x;
            border-color: #245580;
            text-shadow: 0 -1px 0 rgba(0,0,0,.2);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.15),0 1px 1px rgba(0,0,0,.075);
            padding: 5px 10px;
            font-size: 12px;
            line-height: 1.5;
            border-radius: 3px;
            color: #fff;
            background-color: #337ab7;
            display: inline-block;
            margin-bottom: 0;
            font-weight: 400;
            text-align: center;
            white-space: nowrap;
            vertical-align: middle;
            }
        </style>
        <div id="groupsio_embed_signup">
            <form action="https://groups.io/g/hapsira-dev/signup?u=3300935082504621143" method="post" id="groupsio-embedded-subscribe-form" name="groupsio-embedded-subscribe-form" target="_blank">
                <div id="groupsio_embed_signup_scroll">
                    <label for="email" id="templateformtitle">
                        Subscribe to our group
                    </label>
                    <input type="email" value="" name="email" class="email" id="email" placeholder="Email Address" required="">
                    <div style="position: absolute; left: -5000px;" aria-hidden="true">
                        <input type="text" name="b_3300935082504621143" tabindex="-1" value="">
                    </div>
                    <div id="templatearchives"></div>
                    <input type="submit" value="Subscribe" name="subscribe" id="groupsio-embedded-subscribe" class="button">
                </div>
            </form>
        </div>
    </div>
```

```{figure} _static/molniya.png
---
align: right
width: 300px
alt: Molniya Orbit
---
Plot of a [Molniya orbit](https://en.wikipedia.org/wiki/Molniya_orbit) around the Earth <br />
({math}`a = 26600~\mathrm{km}, e = 0.75, i = 63.4\mathrm{^\circ}`).
```

The [source code](https://github.com/pleiszenburg/hapsira) and [issue tracker](https://github.com/pleiszenburg/hapsira/issues) are hosted on GitHub, and all contributions and feedback are more than welcome.

<!-- See [benchmarks](https://benchmarks.poliastro.space/) for the performance analysis of hapsira. -->

`hapsira` works on the recent Python versions and is released under the MIT license, allowing commercial use of the library.

```python
from hapsira.examples import molniya
molniya.plot()
```

## Success stories from the original `poliastro`

> *\"My team and I used Poliastro for our final project in our Summer
> App Space program. This module helped us in plotting asteroids by
> using the data provided to us. It was very challenging finding a
> module that can take orbits from the orbital elements, plot planets,
> and multiple ones. This module helped us because we were able to
> understand the code as most of us were beginners and make some changes
> the way we wanted our project to turn out. We made small changes such
> as taking out the axis and creating a function that will create
> animations. I am happy we used Poliastro because it helped us directs
> us in a direction where we were satisfied of our final product.\"*
>
> \-- Nayeli Ju (2017)
>
> *\"We are a group of students at University of Illinois at
> Urbana-Champaign, United States. We are currently working on a student
> AIAA/AAS satellite competition to design a satellite perform some
> science missions on asteroid (469219) 2016 HO3. We are using your
> hapsira python package in designing and visualizing the trajectory
> from GEO into asteroid's orbit. Thank you for your work on hapsira,
> especially the APIs that are very clear and informational, which helps
> us significantly.\"*
>
> \-- Yufeng Luo (University of Illinois at Urbana-Champaign, United
> States, 2017)
>
> *\"We, at the Institute of Space and Planetary Astrophysics (ISPA,
> University of Karachi), are using Poliastro as part of Space Flight
> Dynamics Text Book development program. The idea is to develop a book
> suitable for undergrad students which will not only cover theoretical
> background but will also focus on some computational tools. We chose
> Poliastro as one of the packages because it was very well written and
> provided results with good accuracy. It is especially useful in
> covering some key topics like the Lambert\'s problem. We support the
> use of Poliastro and open source software because they are easily
> accessible to students (without any charges, unlike some other tools).
> A great plus point for Poliastro is that it is Python based and Python
> is now becoming a very important tool in areas related to Space
> Sciences and Technologies.\"*
>
> \-- Prof. Jawed iqbal, Syed Faisal ur Rahman (ISPA, University of
> Karachi, 2016)

------------------------------------------------------------------------

## Contents

```{toctree}
---
maxdepth: 2
caption: Tutorials
---
installation
quickstart
```

```{toctree}
---
maxdepth: 2
caption: How-to guides & Examples
---
gallery
contributing
core
```

```{toctree}
---
maxdepth: 2
caption: Reference
---
api
bibliography
changelog
```

```{toctree}
---
maxdepth: 2
caption: Background
---
history
related
background
```

```{note}
Older versions of `poliastro`, from which `hapsira` was forked, relied on some Fortran subroutines written by David A. Vallado for his book \"Fundamentals of Astrodynamics and Applications\" and available on the Internet as the [companion software of the book](http://celestrak.com/software/vallado-sw.asp). The author explicitly gave permission to redistribute these subroutines in this project under a permissive license.
```
