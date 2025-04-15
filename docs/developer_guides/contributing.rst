.. _contributing:

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at the `GitHub Issues page`_.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Orion could always use more documentation, whether as part of the
official Orion docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at the `GitHub Issues page`_.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `Orion` for local development.

1. Fork the `Orion` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/Orion.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development::

    $ mkvirtualenv Orion
    $ cd Orion/
    $ make install-develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Try to use the naming scheme of prefixing your branch with ``gh-X`` where X is
   the associated issue, such as ``gh-3-fix-foo-bug``. And if you are not
   developing on your own fork, further prefix the branch with your GitHub
   username, like ``githubusername/gh-3-fix-foo-bug``.

   Now you can make your changes locally.

5. While hacking your changes, make sure to cover all your developments with the required
   unit tests, and that none of the old tests fail as a consequence of your changes.
   For this, make sure to run the tests suite and check the code coverage::

    $ make lint       # Check code styling
    $ make test       # Run the tests
    $ make coverage   # Get the coverage report

6. When you're done making changes, check that your changes pass all the styling checks and
   tests, including other Python supported versions, using::

    $ make test-all

7. Make also sure to include the necessary documentation in the code as docstrings following
   the `Google docstrings style`_.
   If you want to view how your documentation will look like when it is published, you can
   generate and view the docs with this command::

    $ make view-docs

8. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

9. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. It resolves an open GitHub Issue and contains its reference in the title or
   the comment. If there is no associated issue, feel free to create one.
2. Whenever possible, it resolves only **one** issue. If your PR resolves more than
   one issue, try to split it in more than one pull request.
3. The pull request should include unit tests that cover all the changed code
4. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the documentation in an appropriate place.
5. The pull request should work for all the supported Python versions. Check the `GitHub Actions
   Status page`_ and make sure that all the checks pass.

Unit Testing Guidelines
-----------------------

All the Unit Tests should comply with the following requirements:

1. Unit Tests should be based only in unittest and pytest modules.

2. The tests that cover a module called ``orion/path/to/a_module.py``
   should be implemented in a separated module called
   ``tests/orion/path/to/test_a_module.py``.
   Note that the module name has the ``test_`` prefix and is located in a path similar
   to the one of the tested module, just inside the ``tests`` folder.

3. Each method of the tested module should have at least one associated test method, and
   each test method should cover only **one** use case or scenario.

4. Test case methods should start with the ``test_`` prefix and have descriptive names
   that indicate which scenario they cover.
   Names such as ``test_some_methed_input_none``, ``test_some_method_value_error`` or
   ``test_some_method_timeout`` are right, but names like ``test_some_method_1``,
   ``some_method`` or ``test_error`` are not.

5. Each test should validate only what the code of the method being tested does, and not
   cover the behavior of any third party package or tool being used, which is assumed to
   work properly as far as it is being passed the right values.

6. Any third party tool that may have any kind of random behavior, such as some Machine
   Learning models, databases or Web APIs, will be mocked using the ``mock`` library, and
   the only thing that will be tested is that our code passes the right values to them.

7. Unit tests should not use anything from outside the test and the code being tested. This
   includes not reading or writing to any file system or database, which will be properly
   mocked.

Tips
----

To run a subset of tests::

    $ python -m pytest tests.test_orion
    $ python -m pytest -k 'foo'

Release Workflow
----------------

The process of releasing a new version involves several steps combining both ``git`` and
``bumpversion`` which, briefly:

1. Merge what is in ``master`` branch into ``stable`` branch.
2. Update the version in ``setup.cfg``, ``orion/__init__.py`` and
   ``HISTORY.md`` files.
3. Create a new git tag pointing at the corresponding commit in ``stable`` branch.
4. Merge the new commit from ``stable`` into ``master``.
5. Update the version in ``setup.cfg`` and ``orion/__init__.py``
   to open the next development iteration.

.. note:: Before starting the process, make sure that ``HISTORY.md`` has been updated with a new
          entry that explains the changes that will be included in the new version.
          Normally this is just a list of the Pull Requests that have been merged to master
          since the last release.

Once this is done, run of the following commands:

1. If you are releasing a patch version::

    make release

2. If you are releasing a minor version::

    make release-minor

3. If you are releasing a major version::

    make release-major

.. _GitHub issues page: https://github.com/sintel-dev/Orion/issues
.. _GitHub Actions Status page: https://github.com/sintel-dev/Orion/actions
.. _Google docstrings style: https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments

Contributing to CAP
=================

Thank you for your interest in contributing to CAP! This guide will help you get started with contributing to the project.

Getting Started
-------------

1. **Fork the Repository**:
   - Fork the CAP repository on GitHub
   - Clone your fork locally
   - Set up the development environment

2. **Development Environment**:
   .. code-block:: bash

       # Clone your fork
       git clone https://github.com/your-username/CAP.git
       cd CAP

       # Create virtual environment
       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

       # Install development dependencies
       pip install -e ".[dev]"

3. **Code Style**:
   - Follow PEP 8 guidelines
   - Use Black for code formatting
   - Use isort for import sorting
   - Use flake8 for linting

   .. code-block:: bash

       # Format code
       black .
       isort .

       # Run linter
       flake8

Development Workflow
-----------------

1. **Create a Branch**:
   .. code-block:: bash

       # Create and switch to a new branch
       git checkout -b feature/your-feature-name

2. **Make Changes**:
   - Write your code
   - Add tests
   - Update documentation
   - Follow the code style guidelines

3. **Run Tests**:
   .. code-block:: bash

       # Run all tests
       pytest

       # Run specific test file
       pytest tests/test_specific.py

       # Run with coverage
       pytest --cov=cap

4. **Commit Changes**:
   .. code-block:: bash

       # Stage changes
       git add .

       # Commit with a descriptive message
       git commit -m "Description of changes"

5. **Push Changes**:
   .. code-block:: bash

       # Push to your fork
       git push origin feature/your-feature-name

6. **Create Pull Request**:
   - Go to GitHub
   - Create a pull request from your branch
   - Fill in the PR template
   - Request review

Code Organization
---------------

1. **Project Structure**:
   .. code-block:: text

       CAP/
       ├── cap/                # Main package
       │   ├── models/         # Forecasting models
       │   ├── data/          # Data handling
       │   ├── training/      # Training utilities
       │   ├── metrics/       # Evaluation metrics
       │   └── utils/         # Common utilities
       ├── tests/             # Test files
       ├── docs/              # Documentation
       ├── examples/          # Example notebooks
       └── setup.py           # Package setup

2. **Adding New Features**:
   - Place code in appropriate module
   - Add tests in corresponding test file
   - Update documentation
   - Add examples if needed

3. **Adding New Models**:
   - Create new file in `cap/models/`
   - Inherit from `BaseModel`
   - Implement required methods
   - Add tests in `tests/models/`
   - Update documentation

Testing
-------

1. **Test Structure**:
   .. code-block:: text

       tests/
       ├── models/            # Model tests
       ├── data/             # Data tests
       ├── training/         # Training tests
       ├── metrics/          # Metrics tests
       └── utils/            # Utilities tests

2. **Writing Tests**:
   - Use pytest framework
   - Follow AAA pattern (Arrange, Act, Assert)
   - Use fixtures for common setup
   - Add edge cases
   - Use meaningful test names

   .. code-block:: python

       def test_model_prediction():
           # Arrange
           model = Transformer()
           data = create_test_data()

           # Act
           predictions = model.predict(data)

           # Assert
           assert predictions.shape == expected_shape
           assert not np.isnan(predictions).any()

3. **Test Coverage**:
   - Aim for high test coverage
   - Include unit tests
   - Add integration tests
   - Test error cases

Documentation
-----------

1. **Documentation Structure**:
   .. code-block:: text

       docs/
       ├── user_guides/      # User documentation
       ├── developer_guides/ # Developer documentation
       ├── api/             # API reference
       └── examples/        # Example documentation

2. **Writing Documentation**:
   - Use reStructuredText format
   - Follow Google docstring style
   - Include examples
   - Keep documentation up to date

3. **Building Documentation**:
   .. code-block:: bash

       # Install documentation dependencies
       pip install -e ".[docs]"

       # Build documentation
       cd docs
       make html

4. **Documentation Best Practices**:
   - Write clear and concise docs
   - Include code examples
   - Add type hints
   - Document exceptions
   - Keep API reference updated

Code Review Process
-----------------

1. **Pull Request Requirements**:
   - All tests must pass
   - Code must be formatted
   - Documentation must be updated
   - No merge conflicts

2. **Review Checklist**:
   - Code quality
   - Test coverage
   - Documentation
   - Performance impact
   - Backward compatibility

3. **Review Process**:
   - Address review comments
   - Update PR as needed
   - Request re-review if necessary
   - Squash commits if requested

Release Process
-------------

1. **Version Management**:
   - Follow semantic versioning
   - Update version in setup.py
   - Update CHANGELOG.md
   - Create release notes

2. **Release Steps**:
   .. code-block:: bash

       # Update version
       # Update CHANGELOG.md
       # Create release branch
       git checkout -b release/vX.Y.Z

       # Build and test
       python setup.py sdist bdist_wheel
       twine check dist/*

       # Create GitHub release
       # Merge to main
       git checkout main
       git merge release/vX.Y.Z

3. **Release Checklist**:
   - All tests passing
   - Documentation updated
   - CHANGELOG updated
   - Version bumped
   - Release notes created

Getting Help
----------

1. **Communication Channels**:
   - GitHub Issues
   - Pull Requests
   - Discussion Forum

2. **Asking for Help**:
   - Be specific about the problem
   - Provide context
   - Include error messages
   - Share relevant code

3. **Community Guidelines**:
   - Be respectful
   - Follow code of conduct
   - Help others
   - Share knowledge

For more information about CAP's architecture, see the :ref:`architecture` guide.
