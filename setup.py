from setuptools import setup, find_packages, Extension
import os
import sys

# Cố gắng import Cython để biên dịch tăng tốc, nếu không có thì bỏ qua
try:
    from Cython.Build import cythonize
    # Chỉ biên dịch nếu môi trường có sẵn trình biên dịch C
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

ext_modules = []
if USE_CYTHON:
    try:
        ext_modules = cythonize(
            [Extension("pandatools.accessor", ["pandatools/accessor.py"])],
            compiler_directives={'language_level': "3"}
        )
    except Exception:
        # Nếu biên dịch lỗi (thiếu compiler), reset lại để cài dạng Python thuần
        ext_modules = []

setup(
    name="pandatools",
    version="2.0.2",
    author="Sơn Lê",
    description="SƠN AI DataFrame Cleaner",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        "pandas>=2.0",
        "numpy",
    ],
    extras_require={
        "ai": ["openai", "google-generativeai"],
    },
    python_requires=">=3.8",
)