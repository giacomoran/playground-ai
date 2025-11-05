# Disable the fish greeting
set fish_greeting

# Set the default language
# https://unix.stackexchange.com/q/87745
# https://unix.stackexchange.com/q/576701
set -gx LANG en_US.UTF-8
set -gx GITHUB_USER giacomorandazzo

# Fix TERM environment variable for terminals not recognized on remote servers
# When using ghostty or other modern terminals via SSH, the TERM might not be recognized
# This checks if the terminal type is supported and falls back to xterm-256color if not
if test -n "$TERM"
    # Explicitly handle common unrecognized terminal types
    if test "$TERM" = "xterm-ghostty"
        set -gx TERM xterm-256color
    else if type -q infocmp
        # Check if the terminal type is in the terminfo database
        # If infocmp fails, the terminal type is not recognized
        if not infocmp "$TERM" >/dev/null 2>&1
            # Fall back to xterm-256color (widely supported)
            set -gx TERM xterm-256color
        end
    end
end

# Add common binary directories to PATH (for uv, cargo, etc.)
fish_add_path --global ~/.cargo/bin
fish_add_path --global ~/.local/bin

# Use `less` as default pager
# See below for configuration options
set -gx PAGER less

# `less`: Set default options
# -i/--ignore-case         search case-insensitively if pattern is lower-cased
# -R/--RAW-CONTROL-CHARS   display raw control characters for colors and hyperlinks
# -F/--quit-if-one-screen  print instead of paginate if the file is small enough
# -j/--jump-target         specify where to position a target line (either as number of rows or as percentage of screen height)
# -#/--shift               specify how much to scroll horizontally (either as number of cols or as percentage of screen width)
set -gx LESS "-iRFj.2#.2"

# `less`: Don't save history
set -gx LESSHISTFILE -

# Use `bat` to view man pages (or batcat on Ubuntu)
# https://github.com/sharkdp/bat#man
if type -q bat
    set -gx MANPAGER "sh -c 'col -bx | bat -l man -p'"
else if type -q batcat
    set -gx MANPAGER "sh -c 'col -bx | batcat -l man -p'"
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if status is-interactive
    # Aliases
    alias which="type -a"
    alias ls="eza --icons --across"
    alias ll="eza --icons --long --header --git --all"
    alias la="eza --icons --across --all"

    # Abbreviations
    abbr --add --global gs 'git status'
    abbr --add --global ga 'git add'
    abbr --add --global gc 'git commit -m'
    abbr --add --global gl 'git log'
    abbr --add --global gst 'git show-tree' # from git-extras

    # Activate starship
    if type -q starship
        starship init fish | source
    end

    # Activate direnv
    if type -q direnv
        direnv hook fish | source
    end
end
