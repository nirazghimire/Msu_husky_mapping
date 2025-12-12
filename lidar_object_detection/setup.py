from setuptools import find_packages, setup

package_name = 'lidar_object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/detection.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thispc',
    maintainer_email='ng733@msstate.edu',
    description='Lidar object detection using OpenPCDet',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # executable_name = package.module:function
            'lidar_detection_node = lidar_object_detection.detection_node:main',
        ],
    },
)

