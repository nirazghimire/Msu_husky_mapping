import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'basic_mobile_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/' + package_name + '/launch', ['launch/view_robot.launch.py']),
    ('share/' + package_name + '/rviz', ['rviz/urdf_config.rviz']),
    ('share/' + package_name + '/models/urdf', ['models/urdf/basic_mobile_bot_v1.urdf']),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Niraj',
    maintainer_email='ng733@msstate.edu',
    description='basic mobile robot for gazebo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
