from setuptools import setup, find_packages

from ForensicHub.version import __version__


"""
For documentations, see here:
https://github.com/pypa/sampleproject/blob/db5806e0a3204034c51b1c00dde7d5eb3fa2532e/setup.py

在工作路径下运行下列指令：
pip install -e .  
即可实现本地安装本包，并在更新文件时自动更新相应内容，便于调试开发。
"""

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

def requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements

# exit(0)

setup(
    name='forensichub',
    version=__version__,
    description="A codebase integrated Image manipulation detectoin & localization, Deepfake detection, Document manipulation detection and AIGC detection.",
    long_description=readme(),
    long_description_content_type='text/markdown',
    url="https://github.com/scu-zjz/ForensicHub",
    author="Xiaochen Ma",   # Optional
    author_email="xiaochen.ma.cs@gmail.com", # Optional
    

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        
        # Pick your license as you wish
        "License :: Free For Educational Use",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=[
        "AI",
        "artificial intelligence",
        "image forensics",
        "image manipulation localization",
        "image manipulation detection"
        "Deepfake detection",
        "AIGC detection",
        ],  # Optional
    
    packages=find_packages(),
    python_requires=">=3.7, <4",
    

    include_package_data=True, #
    install_requires=requirements(),
    license='CC-BY-4.0',
    entry_points={
        'console_scripts': [
            'forhub=ForensicHub.cli:main',  # cli是你的命令行脚本的模块，main是入口函数
        ],
    },
    project_urls={  # Optional
        "Github": "https://github.com/scu-zjz/ForensicHub/",
        "Documentation":"https://github.com/scu-zjz/ForensicHub-doc",
        "Bug Reports": "https://github.com/scu-zjz/ForensicHub/issues",
    },
)
