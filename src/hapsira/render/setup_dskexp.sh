#!/bin/bash

DSKEXP_URL="https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Linux_64bit/dskexp"

# Functions
install_wget_arch() {
    echo "Installing wget on Arch Linux..."
    sudo pacman -Syu wget --noconfirm
    if [ $? -ne 0 ]; then
        echo "Failed to install wget. Please install it manually and re-run the script."
        exit 1
    fi
}

install_wget_fedora() {
    echo "Installing wget on Fedora..."
    sudo dnf install wget -y
    if [ $? -ne 0 ]; then
        echo "Failed to install wget. Please install it manually and re-run the script."
        exit 1
    fi
}

install_wget_ubuntu() {
    echo "Installing wget on Ubuntu..."
    sudo apt-get update
    sudo apt-get install wget -y
    if [ $? -ne 0 ]; then
        echo "Failed to install wget. Please install it manually and re-run the script."
        exit 1
    fi
}

# wget installations
if ! command -v wget > /dev/null 2>&1; then
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case "$ID" in
            ubuntu|debian)
                install_wget_ubuntu
                ;;
            arch)
                install_wget_arch
                ;;
            fedora)
                install_wget_fedora
                ;;
            *)
                echo "Unsupported operating system. Please install wget manually and re-run the script."
                exit 1
                ;;
        esac
    fi
fi


# dskexp setup
if [ ! -f "dskexp" ]; then
    echo "dskexp not found. Downloading..."
    wget $DSKEXP_URL -O dskexp
    if [ $? -ne 0 ]; then
        echo "Failed to download dskexp. Please check the URL and your internet connection."
        exit 1
    fi
fi

chmod +x dskexp