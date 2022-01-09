
### Using dynamite with docker

To build the docker image, in the root directory of the dynamite git repository run:

```
docker build --build-arg GIT_BRANCH=(git rev-parse --abbrev-ref HEAD) --build-arg GIT_COMMIT=(git describe --always) -t gdmeyer/dynamite:0.1.0-jupyter -f docker/Dockerfile-cpu .
```

To build an image without jupyter, run
```
docker build --target=build --build-arg GIT_BRANCH=(git rev-parse --abbrev-ref HEAD) --build-arg GIT_COMMIT=(git describe --always) -t gdmeyer/dynamite:0.1.0-jupyter -f docker/Dockerfile-cpu .
```

To run a Python script in the current directory with dynamite, do:

```
docker run -it --rm -v "$PWD":/home/dnm/src gdmeyer/dynamite:0.1.0 python your_script.py
```

For example, to run the unit tests in the docker container, you can do

```
cd tests/unit/
docker run -it --rm -v "$PWD":/home/dnm/src gdmeyer/dynamite:0.1.0 python -m unittest discover .
```
