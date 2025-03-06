#!/bin/bash

# Exit on any error
set -e

echo "==== Mininet Installation Script for Ubuntu ===="
echo "Installing as root user..."

# Update package repositories
echo "Updating package repositories..."
apt-get update -y

# Install dependencies
echo "Installing dependencies..."
apt-get install -y git python3-pip python3-matplotlib python3-seaborn python3-numpy build-essential \
                   python3-dev python-is-python3 python3-wheel pylint help2man \
                   pyflakes3 python3-setuptools ethtool iproute2 telnet \
                   net-tools iputils-ping openssh-client openssh-server wireshark tcpdump

# Clone Mininet repository
echo "Cloning Mininet repository from GitHub..."
cd /opt
if [ -d "mininet" ]; then
    echo "Mininet directory already exists. Removing it..."
    rm -rf mininet
fi
git clone https://github.com/mininet/mininet.git

# Go to mininet directory
cd mininet

# Install Mininet with all components
echo "Installing Mininet with all components..."
./util/install.sh -a

# Verify installation
echo "Verifying Mininet installation..."
# Check if mn command is available
if command -v mn &> /dev/null; then
    echo "Mininet command 'mn' is available."
else
    echo "ERROR: Mininet command 'mn' not found. Installation may have failed."
    exit 1
fi

# Test Mininet
echo "Running a simple Mininet test..."
mn --test pingall

# Install additional Python packages required for the script
echo "Installing additional Python packages..."
pip3 install matplotlib seaborn numpy

# Create a symbolic link if needed
echo "Creating symbolic link for python3..."
if [ ! -f /usr/bin/python ]; then
    ln -s /usr/bin/python3 /usr/bin/python
fi

# Open the directory
echo "Opening Mininet directory..."
cd /opt/mininet
echo "Current directory: $(pwd)"
ls -la

echo ""
echo "==== Debugging Commands ===="
echo "1. Check Mininet version: mn --version"
echo "2. Test minimal topology: sudo mn"
echo "3. Test network connectivity: sudo mn --test pingall"
echo "4. Test bandwidth: sudo mn --test iperf"
echo "5. Start Mininet CLI: sudo mn --topo single,3"
echo "6. Check Python version: python --version"
echo "7. Check package installation: pip3 list | grep -E 'matplotlib|seaborn|numpy'"
echo "8. Test running the load balancer script: sudo python /path/to/your/script.py"
echo ""
echo "==== Installation Complete ===="
echo "Mininet has been installed successfully in /opt/mininet"
echo "You can now run your load balancer simulation script."
