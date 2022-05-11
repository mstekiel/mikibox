Compiling and setting up the documentation.

For windows run ./make.bat html in the terminal.

Some of the build directories have been change in order to be able to host the documentation on GitHub Pages.
the main thing is the redirecting index.html file in the main folder, that redirects to the real index file.
this idea is from the blog post:

https://python.plainenglish.io/how-to-host-your-sphinx-documentation-on-github-550254f325ae

Based on the thread:
https://github.com/sphinx-doc/sphinx/issues/3382

From which I also incorporated the idea of keeping the build in the build folder, and adding the empty .nojekyll file to prevent GitHub from running it.