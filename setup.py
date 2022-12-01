from setuptools import setup

setup(name='mikibox',
    version='0.5.1',
    description='Various tools I have been using for data analysis.',
    url='https://github.com/mstekiel/mikibox',
    author='Michal Stekiel',
    author_email='michal.stekiel@gmail.com',
    packages=['mikibox'],
    python_requires='==3.9',
    install_requires=[
        'numpy==1.21.5',
        'matplotlib==3.5.2',
        'scipy==1.9.1'
    ],
    include_package_data=True,
    zip_safe=False)