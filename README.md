[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center">Spectrum</h1>
  
  <p align="center">
    Spectrum is an AI that uses deep learning to generate rap song lyrics.
    <br />
    <br />
    <a href="https://github.com/YigitGunduc/Spectrum">View Demo</a>
    <br />
    <a href="https://github.com/YigitGunduc/Spectrum/issues">Report Bug</a>
    <br />
    <a href="https://github.com/YigitGunduc/Spectrum/issues">Request Feature</a>
  </p>
</p>

<!-- ABOUT THE PROJECT -->
## About The Project

Spectrum is an AI that uses deep learning to generate rap song lyrics.
### Built With

This project is built using Python, Tensorflow, and Flask.

* [Python](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

This section explains and goes thought how to install and setup everything
* pip
```sh
pip install -r requirements.txt
```

### Installation

1. Clone the repo
```sh
git clone https://github.com/YigitGunduc/Spectrum.git
```

<!-- USAGE EXAMPLES -->
## Usage

### Training

Navigate to the spectrum folder
```sh
cd Spectrum/AI
```
Run train.py
```sh
python3 train.py
```
### Generating Text from model
Call eval.py from the command line with seed text as an argument
```sh
python3 eval.py SEEDTEXT 
```

### API
spectrum has a free web API you can send request to it as shown below

```python
import requests 

response = requests.get("http://spectrum.herokuapp.com/api/generate/SEEDTEXT")
#raw response
print(response.json())
#cleaned up response
print(response.json()["lyrics"])
```

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/YigitGunduc/Spectrum/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



[contributors-shield]: https://img.shields.io/github/contributors/YigitGunduc/Spectrum.svg?style=flat-rounded
[contributors-url]: https://github.com/YigitGunduc/Spectrum/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/YigitGunduc/Spectrum.svg?style=flat-rounded
[forks-url]: https://github.com/YigitGunduc/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/YigitGunduc/Spectrum.svg?style=flat-rounded
[stars-url]: https://github.com/YigitGunduc/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/YigitGunduc/Spectrum.svg?style=flat-rounded
[issues-url]: https://github.com/YigitGunduc/Spectrum/issues
[license-url]: https://github.com/YigitGunduc/Spectrum/blob/master/LICENSE
[license-shield]: https://img.shields.io/github/license/YigitGunduc/Spectrum.svg?style=flat-rounded