REMOTE="fdcl-zbook"

echo "Receiving data; press [CTRL+C] to stop..."
while :
do
    rsync -rvuI ${REMOTE}:~/github/ftc/ftc-python/active-ftc-backstepping/data/. ~/github/ftc/ftc-python/active-ftc-backstepping/data/.  --exclude '*.h5' > /dev/null 2>&1  # suppress output
    sleep 1
done
