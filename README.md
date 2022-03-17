# Epidemic Modeling Tool

This project provides epidemic modeling tools beyond what is generally 
available. These tools will focus on providing a geographical interface
with epidemic data, while generating a projected forecast for epidemic
spread.


## Submodules

For this project, we use the John Hopkin's University's dataset for COVID-19
data in the US. The dataset is uploaded to a GitHub repository, which we have
included as a submodule to this repository.

Link: https://github.com/CSSEGISandData/COVID-19

The first time using the submodule:

```git submodule init```

To retrieve new data:

```git submodule update --remote```


## Dependecies

- Python 3.10
- pandas
- matplotlib
- plotly
- dash

## Set-Up

First, make sure you have Python 3.10.

```python3 --version```

If the version is less than 3.10, download the correct version from
here: https://www.python.org/downloads/

Setup a python virtual environment in the project folder.

```python3 -m venv env```

Next, activate the virtual environment.

```
source env/bin/activate   # Unix or MacOS
env/Scripts/activate      # Windows
```

If you are using Windows Powershell, you may have to update permissions
before activating the environment for the first time.

```Set-ExecutionPolicy RemoteSigned```

Update the pip package installer.

```pip install --upgrade pip```

Then, download the required dependencies.

```pip install pandas matplotlib plotly dash diskcache multiprocess psutil```

You can then deactive the virtual environment.

```deactivate```


## Usage

First, activate the virtual environment.

```
source env/bin/activate   # Unix or MacOS
env/Scripts/activate      # Windows
```

Then, run the python script.

```python src/main.py```

## Troubleshooting

If you are on OSX and are recieving a `SSL: CERTIFICATE_VERIFY_FAILED` error, run
the following command:

```/Applications/Python\ 3.6/Install\ Certificates.command```
