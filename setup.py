from setuptools import setup, find_packages

setup(name="keithley2600",
      version="1.1.0",
      description="Full Python driver for the Keithley 2600 series.",
      url="https://github.com/OE-FET/keithley2600.git",
      author="Sam Schott",
      author_email="ss2151@cam.ac.uk",
      licence='MIT',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=[
          'PyVISA',
          'pyvisa-py>=0.3.1',
          'setuptools',
          'numpy',
          'matplotlib',
      ],
      zip_safe=False,
)
