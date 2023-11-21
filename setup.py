from setuptools import setup, find_packages
print('READ README')
setup(
    name='Pseudo-DXA',  
    version='1.0', 
    description='A model for generating pseudo-DXA images from 3D meshes using deep learning .',  
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
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  
)

