# Base Architectures

Some models can be reusable as Unet.
Several unet could be built using different convolution or blocks.
However, the final model will still be a UNet.

In the ```base_architectures``` folder, we intend to provide base architecture builder which could be used across tasks and datasets.

We provide two UNet implementations:

* ```UnetBasedModel```
* ```UnwrappedUnetBasedModel```

The main difference between them if ```UnetBasedModel``` implements the forward function and ```UnwrappedUnetBasedModel``` doesn't.

<h4> UnetBasedModel </h4>
```python
    def forward(self, data):
        if self.innermost:
            data_out = self.inner(data)
            data = (data_out, data)
            return self.up(data)
        else:
            data_out = self.down(data)
            data_out2 = self.submodule(data_out)
            data = (data_out2, data)
            return self.up(data)
```

The UNet will be built recursively from the middle using the ```UnetSkipConnectionBlock``` class.

<h4> UnetSkipConnectionBlock </h4>
```python
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|

    """
```
<h4> UnwrappedUnetBasedModel </h4>

The ```UnwrappedUnetBasedModel``` will create the model based on the configuration and add the created layers within the followings ```ModuleList```

```python
  self.down_modules = nn.ModuleList()
  self.inner_modules = nn.ModuleList()
  self.up_modules = nn.ModuleList()
```
