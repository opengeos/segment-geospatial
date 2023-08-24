# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/giswqs/segment-geospatial/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with `bug` and
`help wanted` is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
`enhancement` and `help wanted` is open to whoever wants to implement it.

### Write Documentation

segment-geospatial could always use more documentation,
whether as part of the official segment-geospatial docs,
in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/giswqs/segment-geospatial/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to implement.
-   Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up segment-geospatial for local development.

1.  Fork the segment-geospatial repo on GitHub.

2.  Clone your fork locally:

    ```shell
    $ git clone git@github.com:your_name_here/segment-geospatial.git
    ```

3.  Install your local copy into a virtualenv. Assuming you have
    virtualenvwrapper installed, this is how you set up your fork for
    local development:

    ```shell
    $ mkvirtualenv segment-geospatial
    $ cd segment-geospatial/
    $ python setup.py develop
    ```

4.  Create a branch for local development:

    ```shell
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

5.  When you're done making changes, check that your changes pass flake8
    and the tests, including testing other Python versions with tox:

    ```shell
    $ flake8 segment-geospatial tests
    $ python setup.py test or pytest
    $ tox
    ```

    To get flake8 and tox, just pip install them into your virtualenv.

6.  Commit your changes and push your branch to GitHub:

    ```shell
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

7.  Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests (see the section below - [Unit Testing](https://samgeo.gishub.org/contributing/#unit-testing)).
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    the feature to the list in README.md.
3.  The pull request should work for Python 3.8, 3.9, 3.10, and 3.11. Check <https://github.com/giswqs/segment-geospatial/pull_requests> and make sure that the tests pass for all
    supported Python versions.

## Unit Testing

Unit tests are in the `tests` folder. If you add new functionality to the package, please add a unit test for it. You can either add the test to an existing test file or create a new one. For example, if you add a new function to `samgeo/samgeo.py`, you can add the unit test to `tests/test_samgeo.py`. If you add a new module to `samgeo/<MODULE-NAME>`, you can create a new test file in `tests/test_<MODULE-NAME>`. Please refer to `tests/test_samgeo.py` for examples. For more information about unit testing, please refer to this tutorial - [Getting Started With Testing in Python](https://realpython.com/python-testing/).

To run the unit tests, navigate to the root directory of the package and run the following command:

```bash
python -m unittest discover tests/
```

## Add new dependencies

If you PR involves adding new dependencies, please make sure that the new dependencies are available on both [PyPI](https://pypi.org/) and [conda-forge](https://conda-forge.org/). Search [here](https://conda-forge.org/feedstock-outputs/) to see if the package is available on conda-forge. If the package is not available on conda-forge, it can't be added as a required dependency in `requirements.txt`. Instead, it should be added as an optional dependency in `requirements_dev.txt`.

If the package is available on PyPI and conda-forge, but if it is challenging to install the package on some operating systems, we would recommend adding the package as an optional dependency in `requirements_dev.txt` rather than a required dependency in `requirements.txt`.

The dependencies required for building the documentation should be added to `requirements_docs.txt`. In most cases, contributors do not need to add new dependencies to `requirements_docs.txt` unless the documentation fails to build due to missing dependencies.
