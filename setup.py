from setuptools import setup
from setuptools import find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

install_requires = [r for r in required if "==" in r]
install_requires[-1] = install_requires[-1].split(";")[0]
dependency_links = [r[7:] for r in required if "-e git+" in r]
install_requires += [l.split("=")[-1] for l in dependency_links]

setup(
    name="torch-points3d",
    version="0.1",
    description="Torch-Point3D: A Unifying Framework for Processing Point Clouds with Deep Learning",
    url="https://github.com/nicolas-chaulet/torch_points3d",
    author="Thomas Chaton, Nicolas Chaulet",
    author_email="thomas.chaton.ai@gmail.com, nicolas.chaulet@gmail.com",
    license="MIT",
    install_requires=install_requires,
    dependency_links=dependency_links,
    python_requires=">=3.6",
    packages=find_packages(),
    zip_safe=False,
)
