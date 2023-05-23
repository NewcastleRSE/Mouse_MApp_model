## About

The code use to process the images and run the Mouse MApp model.

### Project Team
* Matt Leach - ([matthew.leach@newcastle.ac.uk](mailto:matthew.leach@newcastle.ac.uk))
* Satnam Dlay - ([satnam.dlay@newcastle.ac.uk](mailto:rsatnam.dlay@newcastle.ac.uk))
* Fiona Galston


### RSE Contact
Nik Khadijah Nik Aznan
RSE Team  
Newcastle University  
([nik.nik-aznan@newcastle.ac.uk](mailto:nik.nik-aznan@newcastle.ac.uk))  

### Built With

[Python=3.10.7](https://www.python.org/downloads/)  

### How to use

To run the inference, copy the model weight into model folder.
Run ``` python src/inference_pytorch.py```

For the tensorflow version (the model we used for the apps)

[tensorflowjs=3.15.0](https://www.tensorflow.org/js/tutorials/setup)

[keras=2.8.0 ](https://www.tensorflow.org/install)

For the pytorch version, run `python train_pytorch.py --model alexnet`

[pytorch=1.12.1](https://pytorch.org/)

### Data
For the images, please contact Matt Leach - ([matthew.leach@newcastle.ac.uk](mailto:matthew.leach@newcastle.ac.uk)).
The csv files for this project can be found in [Mouse MApp] (https://newcastle.sharepoint.com/:f:/r/sites/rseteam/Shared%20Documents/General/Projects/Mouse%20MApp?csf=1&web=1&e=GOh8eo). The filename has been processed to remove the whitespace and characters such as `(` and `)`.
