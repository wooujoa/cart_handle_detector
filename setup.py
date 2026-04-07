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
            'blue_rect_detector = cart_handle_detector.blue_rect_detector:main',
            'hsv_debug = cart_handle_detector.hsv_debug:main',
            'cart_handle_detect = cart_handle_detector.cart_handle_detect:main',
            'blue_dot_zed_calib = cart_handle_detector.blue_dot_zed_calib:main',
            'cart_detect_pub = cart_handle_detector.cart_detect_pub:main',
            'yellow_cart_detect = cart_handle_detector.yellow_cart_detect:main',
            'yellow_detect = cart_handle_detector.yellow_detect:main',
        ],
    },
)
