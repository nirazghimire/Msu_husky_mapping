import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lidar_mapping_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),[
            f for f in glob('launch/**/**', recursive=True) if os.path.isfile(f)
        ]
         
         )
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Niraj',
    maintainer_email='ng733@msstate.edu',
    description='setup file for launching the launchfile ',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
