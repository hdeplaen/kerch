# coding=utf-8
"""
Script to install PyTorch on Windows and Linux

Automatically detects the available CUDA versions and prompts the user to select one.

Usage:
    install_torch [cpu|gpu]
"""

import os
import re
import stat
import subprocess
import sys
from rich.prompt import Prompt, Confirm
from bs4 import BeautifulSoup as bs
from requests_html import HTMLSession
import wget

session = HTMLSession()


def error_message(title, message):
    print(f"\033[91m{title}\033[0m: {message}")


def warning_message(title, message):
    print(f"\033[93m{title}\033[0m: {message}")


def scrape_installer_url(cuda_version):
    installer_url = ""
    cuda_version = cuda_version.replace(".", "-")
    if sys.platform == "win32":
        url = f"https://developer.nvidia.com/cuda-{cuda_version}-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local"

        resp = session.get(url)
        resp.html.render()
        html = resp.html.html

        # page = requests.get(url, timeout=None)
        soup = bs(html, "html.parser")
        installer_url = soup.find("a", {"id": "targetDownloadButtonHref"}).get("href")
        return installer_url

    elif sys.platform == "linux" or sys.platform == "linux2":
        url = f"https://developer.nvidia.com/cuda-{cuda_version}-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local"

        resp = session.get(url)
        resp.html.render()
        html = resp.html.html

        # page = requests.get(url, timeout=None)
        soup = bs(html, "html.parser")
        installer_url = (
            soup.find("span", {"class": "cudaBash"})
            .text.replace("wget ", "")
            .replace("%3A", ":")
        )

    print(installer_url)
    return installer_url


def install_cpu_torch():
    # Install CPU version of PyTorch
    os.system("pip install torch torchvision torchaudio")


def get_available_cuda_versions():
    url = "https://pytorch.org/"

    resp = session.get(url)
    resp.html.render()
    html = resp.html.html
    soup = bs(html, "html.parser")
    version_1 = soup.find("div", {"id": "cuda.x"}).find("div").text
    version_2 = soup.find("div", {"id": "cuda.y"}).find("div").text

    return [ver.replace("CUDA ", "") for ver in [version_1, version_2]]


def install_gpu_torch(cuda_version):
    print(f"Installing PyTorch for CUDA={cuda_version}")
    # Install GPU version of PyTorch
    cuda_installed = False
    if os.system("nvcc --version") != 0:
        cuda_installed = False
        warning_message("Error", "CUDA is not installed")
    else:
        cuda_installed = True
        output_nvcc = subprocess.check_output("nvcc --version", shell=True)
        if cuda_version not in str(output_nvcc):
            cuda_installed = False
            current_version = re.findall(
                r"release (\d+\.\d+)", str(output_nvcc), re.MULTILINE
            )[0]
            warning_message(
                "Error",
                f"CUDA {cuda_version} is not installed. Currently installed version is {current_version}",
            )

    if not cuda_installed:
        if Confirm.ask("Do you want to install CUDA?", default=True):
            download_url = scrape_installer_url(cuda_version)
            wget.download(download_url)

            downloaded_file = os.path.abspath("./" + download_url.split("/")[-1])

            st = os.stat(downloaded_file)
            os.chmod(downloaded_file, st.st_mode | stat.S_IEXEC)
            os.system(downloaded_file)

    cuda_version = cuda_version.replace(".", "")
    os.system(
        f"pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu{cuda_version}"
    )


def main():
    print("PyTorch Installer v0.1")
    print("By: Mikołaj Badyl")
    print(r"Github: https://github.com/hawier-dev")
    print("-----------------------------------------")

    selected_device = Prompt.ask(
        "Select device", choices=["cpu", "gpu"], default="cpu"
    )

    if selected_device == "cpu":
        install_cpu_torch()
    elif selected_device == "gpu":
        versions = get_available_cuda_versions()

        selected_version = Prompt.ask(
            "Select CUDA version", choices=versions, default=versions[-1]
        )
        install_gpu_torch(selected_version)

    else:
        error_message("Error", "Invalid argument")
        print("Usage: python install_torch.py [cpu|gpu]")


if __name__ == "__main__":
    main()