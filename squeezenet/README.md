mkdir build

cd build

cmake ..

make

sudo ./squeezenet -s   // serialize model to plan file i.e. 'squeezenet.engine'
sudo ./squeezenet -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/squeezenet
```

