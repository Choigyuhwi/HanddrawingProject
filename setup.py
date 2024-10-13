from setuptools import setup, find_packages

setup(
    name='handdrawing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # 필요한 패키지를 여기에 나열하세요
    ],
    author='ChoiGyuHwi', # 작성자
    author_email='myckh527@gmail.com', # 이메일
    description='A package for processing and analyzing hand-drawn images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://https://github.com/Choigyuhwi/HanddrawingProject', # 깃허브 주소

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)