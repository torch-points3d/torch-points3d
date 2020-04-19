from setuptools import setup
from setuptools import find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

required = [r for r in required if "==" in r]

setup(
    name="torch-points3d",
    version="0.1",
    description="Torch-Point3D: A Unifying Framework for Processing Point Clouds with Deep Learning",
    url="https://github.com/nicolas-chaulet/torch_points3d",
    author="Thomas Chaton, Nicolas Chaulet",
    author_email="thomas.chaton.ai@gmail.com, nicolas.chaulet@gmail.com",
    license="MIT",
    install_requires=required,
    python_requires=">=3.6",
    packages=find_packages(),
    zip_safe=False,
)
