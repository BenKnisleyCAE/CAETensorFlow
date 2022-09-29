CAE TensorFlow 
=

## How to Build Docker Image
```sh
docker build -t tensorflow .
```


## How to Run Docker Image
```sh
docker run -p 7008:7008 -v /mnt/c/Users/knisleyb/Desktop/tensorflow_mnt/input/:/mnt/input -v /mnt/c/Users/knisleyb/Desktop/tensorflow_mnt/output/:/mnt/output tensorflow
```