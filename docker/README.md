```bash
# under the root directory, run the commands below:
docker build -f docker/Dockerfile .
docker run -p 8888:8888 -p 8675:8675 --name ai-toolkit-dev -e JUPYTER_TOKEN=123 -it {TODO}
```