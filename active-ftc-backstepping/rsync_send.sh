REMOTE="fdcl-zbook"
rsync ~/github/ftc/ftc-python/. -rvuI ${REMOTE}:~/github/ftc/ftc-python/. --exclude data --exclude '*.swp' --exclude '*.sh' --exclude '*.DS_Store'
