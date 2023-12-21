[![hapsira Logo](docs/source/_static/logo_readme.png)](https://hapsira.readthedocs.io/)


# hapsira

**🚀 Astrodynamics in Python - a `poliastro` fork**

[![ci_badge](https://github.com/pleiszenburg/hapsira/actions/workflows/test.yaml/badge.svg?branch=main "Build Status: main / release")](https://github.com/pleiszenburg/hapsira/actions/workflows/test.yaml)
[![docs_badge](https://readthedocs.org/projects/hapsira/badge/?version=latest "Build Status: Docs")](https://hapsira.readthedocs.io/en/latest/?badge=latest)
[![pre_commit_badge](https://results.pre-commit.ci/badge/github/pleiszenburg/hapsira/main.svg "pre-commit badge")](https://results.pre-commit.ci/latest/github/pleiszenburg/hapsira/main)
[![python_badge](https://img.shields.io/pypi/pyversions/hapsira?logo=pypi&logoColor=white "python badge")](https://pypi.org/project/hapsira)
[![pypi_badge](https://img.shields.io/pypi/v/hapsira.svg?logo=Python&logoColor=white?labelColor=blue "pypi badge")](https://pypi.org/project/hapsira)
[![license_badge](https://img.shields.io/badge/license-MIT-blue.svg?logo=open%20source%20initiative&logoColor=white "license badge")](https://opensource.org/licenses/MIT)
[![mailing_badge](https://img.shields.io/badge/mailing%20list-groups.io-8cbcd1.svg "mailing badge")](https://groups.io/g/hapsira-dev)
[![chat_badge](https://img.shields.io/matrix/hapsira:matrix.org.svg?logo=Matrix&logoColor=white "chat badge")](https://matrix.to/#/#hapsira:matrix.org)


## Synopsis

`hapsira` is an open source ([MIT](https://opensource.org/licenses/MIT)) pure Python library for interactive Astrodynamics and Orbital Mechanics, with a focus on ease of use, speed, and quick visualization. It provides a simple and intuitive API, and handles physical quantities with units. Its features include orbit propagation, solution of Lambert\'s problem, conversion between position and velocity vectors and classical orbital elements and orbit plotting, among others. It focuses on interplanetary applications, but can also be used to analyze artificial satellites in Low-Earth Orbit (LEO).

![Multiple examples image](docs/source/_static/examples.png)

This project is a fork of the as of October 2023 unmaintained & archived [poliastro](https://github.com/poliastro/poliastro) package.


## Installation

`hapsira` is supported on Linux, macOS and Windows on Python 3.8 to 3.11. Python 3.8 support will be dropped and Python 3.12 support will be added as soon as [numba 0.59 gets released](https://github.com/numba/numba/issues/9197).

Multiple installation methods are supported by `hapsira`, including:

|                             **Logo**                              | **Platform** |                                    **Command**                                    |
|:-----------------------------------------------------------------:|:------------:|:---------------------------------------------------------------------------------:|
|       ![PyPI logo](https://simpleicons.org/icons/pypi.svg)        |     PyPI     |                        ``python -m pip install hapsira``                        |
| ![Conda Forge logo](https://simpleicons.org/icons/condaforge.svg) | Conda Forge  |                 ``conda install hapsira --channel conda-forge``                 |
|     ![GitHub logo](https://simpleicons.org/icons/github.svg)      |    GitHub    | ``python -m pip install https://github.com/pleiszenburg/hapsira/archive/main.zip`` |

For other installation methods, see the [alternative installation methods](https://hapsira.readthedocs.io/en/stable/installation.html#alternative-installation-methods).


## Migrating from `poliastro` to `hapsira`

`hapsira` is based on `poliastro main` as of the date of its archival, October 14 2023, [commit 1b01768c](https://github.com/poliastro/poliastro/commit/21fd7719e89a7d22b4eac63141a60a7f1b01768c). Therefore, `hapsira`'s initial release already includes a number of changes made since `poliastro`'s last release, `0.17.0`. Most notably, the plotter framework saw a redesign and is backwards incompatible, see [poliastro #1545](https://github.com/poliastro/poliastro/pull/1545). For further details, see [changelog](https://github.com/pleiszenburg/hapsira/blob/main/docs/source/changelog.md).


## Documentation

Complete documentation, including a [quickstart guide](https://hapsira.readthedocs.io/en/stable/quickstart.html) and an [API reference](https://hapsira.readthedocs.io/en/latest/api.html) can be read on [Read the Docs](https://readthedocs.org).


## Examples

There is a great variety of examples demonstrating the capabilities of `hapsira`. Examples can be accessed in various ways:

* Examples source code collected in the [examples directory](https://github.com/pleiszenburg/hapsira/tree/main/docs/source/examples)
* Rendered [gallery of examples](https://hapsira.readthedocs.io/en/latest/gallery.html) presented in the documentation


## Problems and suggestions

If for any reason you get an unexpected error message or an incorrect result, or you want to let the developers know about your use case, please open a new issue in the [issue tracker](https://github.com/pleiszenburg/hapsira/issues) and we will try to answer promptly.


## Contributing and community support

This project exists thanks to all the people who contribute! `hapsira` is a community project, hence all contributions are more than welcome! For more information, head to the [CONTRIBUTING.md](https://github.com/pleiszenburg/hapsira/blob/main/CONTRIBUTING.md) file.

Release announcements and general discussion take place on our [mailing list](https://groups.io/g/hapsira-dev).

For further clarifications, feature suggestions and discussions, feel free to join `hapsira`'s [chat room](https://matrix.to/#/#hapsira:matrix.org).


## Commercial support

Commercial support is provided by [pleiszenburg.de - Independent Scientific Services](https://pleiszenburg.de).


## Frequently asked questions

* **What's up with the name?**

  "hapsira" is a shorthand for "[hapësira kozmike](https://sq.wikipedia.org/wiki/Hap%C3%ABsira_kozmike)", which is Albanian for "space". Why Albanian? Other than virtually every reasonable English (and German for that matter) package name related to this topic already taken, [Wag the Dog](https://en.wikipedia.org/wiki/Wag_the_Dog) is a film definitely worth watching.

* **Why a fork of poliastro?**

  See [here](https://github.com/poliastro/poliastro/issues/1640).

* **Is hapsira validated?**

  `hapsira` is a fork of `poliastro`, [which was validated](https://github.com/poliastro/validation/) against other [commonly used Astrodynamics software](https://hapsira.readthedocs.io/en/stable/related.html) such as GMAT and Orekit. Currently, `hapsira`'s results are matching those of `poliastro`.

* **Can I suggest new features for hapsira?**

  Sure, we encourage you to [open an issue](https://github.com/pleiszenburg/hapsira/issues) so we can discuss future feature additions!

* **What's the future of the project?**

  The best way to get an idea of the roadmap is to see the [milestones](https://github.com/pleiszenburg/hapsira/milestones) of the project.
