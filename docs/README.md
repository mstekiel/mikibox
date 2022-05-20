## Documentation is available at

https://mstekiel.github.io/mikibox/build/html/index.html

It is formatted according to the numopydoc standards and Google Style for Python

https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

https://djangocas.dev/docs/4.0/python-docstring-google-style-example.html

### Compiling and setting up the documentation.

For windows run ./make.bat html in the terminal.

Some of the build directories have been rearranged in order to be able to host the documentation on GitHub Pages.
The main thing is the redirecting index.html file in the main folder, that redirects to the real index file.

This idea is from the blog post:
https://python.plainenglish.io/how-to-host-your-sphinx-documentation-on-github-550254f325ae

Based on the thread:
https://github.com/sphinx-doc/sphinx/issues/3382

From which I also incorporated the idea of keeping the build in the build folder, and adding the empty .nojekyll file to prevent GitHub from running it.