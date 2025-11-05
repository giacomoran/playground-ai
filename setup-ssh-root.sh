#!/bin/bash
set -euo pipefail

# Root setup script for ephemeral GPU servers
# Run this script as root first, then connect as 'giacomo' and run project-specific setup
#
# Assumptions:
# - Ubuntu-based image
# - Python, PyTorch, and CUDA are pre-installed (for ML workloads)
# - You are connecting via SSH key (required for password auth to be disabled)
# - You will edit your local ~/.ssh/config to add the host with:
#     User giacomo
#     ForwardAgent yes

echo "=== Installing sudo (required for subsequent steps) ==="
apt-get update
apt-get install -y sudo

echo ""
echo "=== Creating giacomo user ==="
if id "giacomo" &>/dev/null; then
    echo "User giacomo already exists, skipping creation"
else
    # Create user with bash initially (fish will be installed and set later)
    useradd -m -s /bin/bash giacomo
    usermod -aG sudo giacomo
    echo "User giacomo created with sudo access"
fi

# Add giacomo to docker group if docker is installed (common for GPU servers)
if command -v docker &> /dev/null; then
    usermod -aG docker giacomo || true
    echo "Added giacomo to docker group"
fi

# Copy SSH authorized_keys from root/ubuntu to giacomo so they can connect
# This ensures giacomo can authenticate even after password auth is disabled
SOURCE_KEY=""
if [ -f /root/.ssh/authorized_keys ]; then
    SOURCE_KEY="/root/.ssh/authorized_keys"
elif [ -f /home/ubuntu/.ssh/authorized_keys ]; then
    SOURCE_KEY="/home/ubuntu/.ssh/authorized_keys"
fi

if [ -n "$SOURCE_KEY" ]; then
    sudo -u giacomo mkdir -p /home/giacomo/.ssh
    cp "$SOURCE_KEY" /home/giacomo/.ssh/authorized_keys
    chown giacomo:giacomo /home/giacomo/.ssh/authorized_keys
    chmod 700 /home/giacomo/.ssh
    chmod 600 /home/giacomo/.ssh/authorized_keys
    echo "Copied SSH authorized_keys to giacomo user (from $(dirname $SOURCE_KEY))"
else
    echo "Warning: No authorized_keys found in /root/.ssh or /home/ubuntu/.ssh"
    echo "You may need to manually add your public key to /home/giacomo/.ssh/authorized_keys"
fi

echo ""
echo "=== Installing common tools ==="

# Install all packages via apt-get in one command
apt-get install -y \
    git \
    htop \
    curl \
    wget \
    tmux \
    screen \
    build-essential \
    less \
    ripgrep \
    git-extras \
    fzf \
    fish \
    direnv \
    bat \
    tree \
    gnupg \
    || true

# Configure git for giacomo user
sudo -u giacomo sh -c 'git config --global user.name "Giacomo Randazzo" && git config --global user.email "giacomoran@gmail.com"'
echo "Configured git for giacomo user"

# On Ubuntu, bat is installed as batcat, create symlink so 'bat' works
if command -v batcat &> /dev/null && ! command -v bat &> /dev/null; then
    ln -sf /usr/bin/batcat /usr/local/bin/bat || true
    echo "Created symlink: bat -> batcat"
fi

echo ""
echo "=== Installing starship ==="

# Install starship
if ! command -v starship &> /dev/null; then
    curl -sS https://starship.rs/install.sh | sh
    # Make starship available system-wide (check multiple possible locations)
    STARSHIP_FOUND=false
    for location in "/usr/local/bin/starship" "/root/.local/bin/starship" "/root/.cargo/bin/starship"; do
        if [ -f "$location" ]; then
            # If not already in /usr/local/bin, create symlink
            if [ "$location" != "/usr/local/bin/starship" ]; then
                ln -sf "$location" /usr/local/bin/starship || true
            fi
            echo "Installed starship (found at $location)"
            STARSHIP_FOUND=true
            break
        fi
    done
    # Verify starship is accessible
    if command -v starship &> /dev/null; then
        echo "✓ starship is now available in PATH"
    elif [ "$STARSHIP_FOUND" = false ]; then
        echo "Warning: starship installed but binary not found in expected locations"
        echo "You may need to add starship to PATH manually"
        echo "Try running: which starship"
    fi
else
    echo "starship already installed"
fi

echo ""
echo "=== Installing eza via apt repository ==="

# Install eza via apt repository
if ! command -v eza &> /dev/null; then
    mkdir -p /etc/apt/keyrings
    if wget -qO- https://raw.githubusercontent.com/eza-community/eza/main/deb.asc | gpg --dearmor -o /etc/apt/keyrings/gierens.gpg; then
        echo "deb [signed-by=/etc/apt/keyrings/gierens.gpg] http://deb.gierens.de stable main" | tee /etc/apt/sources.list.d/gierens.list
        chmod 644 /etc/apt/keyrings/gierens.gpg /etc/apt/sources.list.d/gierens.list
        apt-get update
        apt-get install -y eza
        echo "Installed eza"
    else
        echo "Warning: Failed to add eza repository GPG key, skipping eza installation"
    fi
else
    echo "eza already installed"
fi

echo ""
echo "=== Configuring fish shell for giacomo user ==="

# Set fish as default shell for giacomo
if command -v fish &> /dev/null; then
    chsh -s /usr/bin/fish giacomo || true
    echo "Set fish as default shell for giacomo"

    # Download and place fish configuration
    sudo -u giacomo mkdir -p /home/giacomo/.config/fish
    if curl -fsSL https://raw.githubusercontent.com/giacomoran/playground-ai/main/fish-config.fish -o /home/giacomo/.config/fish/config.fish; then
        chown giacomo:giacomo /home/giacomo/.config/fish/config.fish
        echo "Downloaded and configured fish config"
    else
        echo "Warning: Failed to download fish config, you may need to do this manually"
    fi
fi

echo ""
echo "=== Hardening SSH configuration ==="
SSH_CONFIG="/etc/ssh/sshd_config"

# Backup original config
BACKUP_FILE="${SSH_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
if cp "$SSH_CONFIG" "$BACKUP_FILE"; then
    echo "Backed up SSH config to $BACKUP_FILE"
else
    echo "Warning: Failed to backup SSH config, proceeding anyway"
fi

# Apply SSH hardening (only if not already present)
if ! grep -q "# Hardening configuration added by setup-ssh-root.sh" "$SSH_CONFIG" 2>/dev/null; then
    cat >> "$SSH_CONFIG" <<EOF

# Hardening configuration added by setup-ssh-root.sh
PermitRootLogin no
PasswordAuthentication no
EOF
    echo "Applied SSH hardening configuration"
else
    echo "SSH hardening configuration already present, skipping"
fi

# Restart SSH service
if systemctl restart sshd 2>/dev/null || service ssh restart 2>/dev/null; then
    echo "SSH service restarted"
else
    echo "Warning: Failed to restart SSH service, manual restart may be required"
fi

echo ""
echo "=== Setup complete ==="

echo "Next steps:"
echo "1. Add this host to your local ~/.ssh/config with:"
echo "   Host <hostname>"
echo "     HostName <IP_ADDRESS>"
echo "     Port <PORT>"
echo "     User giacomo"
echo "     ForwardAgent yes"
echo ""
echo "2. Connect as giacomo user: ssh -p <PORT> giacomo@<IP_ADDRESS>"
echo "   Or: ssh <hostname>"
echo "3. Run the project-specific setup script"
