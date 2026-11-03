from setuptools import find_packages, setup

package_name = 'moon_rover_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MicahLephart',
    maintainer_email='micahlephart@gmail.com',
    description='Collector and hauler rover vehicle nodes for the Gazebo moon world.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'collector_rover_node = moon_rover_bridge.collector_rover_node:main',
            'hauler_rover_node = moon_rover_bridge.hauler_rover_node:main',
        ],
    },
)
