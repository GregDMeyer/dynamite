
### Building dynamite for Docker

To build the generic dynamite image, in the root directory of the dynamite git repository run:

```
docker build --build-arg HARDWARE=cpu -f docker/Dockerfile --target release -t gdmeyer/dynamite:latest .
```

For the Jupyter version, just change the target to ``jupyter``, and to include GPU support change HARDWARE to "gpu".

To build all relevant images (with and without Jupyter, and cpu/gpu), and with a clean checkout of the git (to avoid including any extraneous files), you can also just run the script ``build.py`` in this directory.
