from glob import glob

from setuptools import find_packages, setup

package_name = 'moon_rover_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config/scenarios', glob('config/scenarios/*.yaml')),
        ('share/' + package_name + '/config/maps', glob('config/maps/*.npz') + glob('config/maps/*.txt')),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='MicahLephart',
    maintainer_email='micahlephart@gmail.com',
    description='Collector and hauler rover vehicle nodes for the Gazebo moon world.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'collector_rover_node = moon_rover_bridge.collector_rover_node:main',
            'hauler_rover_node = moon_rover_bridge.hauler_rover_node:main',
            'collector_mission_node = moon_rover_bridge.collector_mission_node:main',
            'hauler_mission_node = moon_rover_bridge.hauler_mission_node:main',
            'scenario_runner = moon_rover_bridge.scenario_runner:main',
            'build_hazard_map = moon_rover_bridge.build_hazard_map:main',
            'drive_check = moon_rover_bridge.drive_check:main',
        ],
    },
)
