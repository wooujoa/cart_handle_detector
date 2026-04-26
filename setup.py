from setuptools import find_packages, setup

package_name = 'cart_handle_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jwg',
    maintainer_email='wjddnrud4487@kw.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hsv_debug = cart_handle_detector.hsv_debug:main',
            '3d_handle = cart_handle_detector.3d_handle:main',
            'cart_detect = cart_handle_detector.cart_detect:main',
            'feature_detect = cart_handle_detector.feature_detect:main',
            'purple_feature_detect = cart_handle_detector.purple_feature_detect:main',
        ],
    },
)
