from setuptools import setup, find_packages

setup(
    name='Pseudo-DXA',  # Replace with your actual package name if different
    version='1.0',  # Update the version as necessary
    description='A package for generating pseudo-DXA images using machine learning techniques.',  # Update with your package's description
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LambertLeong/Pseudo-DXA',
    author='Lambert Leong',
    author_email='lamberttleong@gmail.com',
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'opencv-python',
        'tensorflow',
        'scikit-learn',  # Assuming use in ML algorithms
        'matplotlib',    # Common for plotting in scientific computing
        'scipy',         # For scientific computing tasks
        'plyfile',       # For PLY file handling
        # Include other necessary packages specific to your project
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Update as per your project's maturity
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        # Add other classifiers as needed
    ],
    python_requires='>=3.6',  # Ensure compatibility with your minimum required Python version
)

