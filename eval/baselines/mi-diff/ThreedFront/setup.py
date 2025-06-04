from setuptools import setup, find_packages


def get_install_requirements():
    return [
        "numpy",
        "pillow",
        "pyrr",
        "pyyaml",
        "torch",
        "torchvision",
        "scipy",
        "tqdm",
        "seaborn",
        "trimesh",
        "simple-3dviz==0.7.0",
    ]


def setup_package():
    setup(
        name="threed_front",
        maintainer="Siyi Hu",
        maintainer_email="siyihu02@gmail.com",
        version="1.0",
        license="BSD-2-Clause",
        description="3D-FRONT dataset classes, preprocessing and evaluation package",
        packages=find_packages(include=["threed_front"]),
        install_requires=get_install_requirements(),
    )

if __name__ == "__main__":
    setup_package()
