<!---
This is a Markdown file, the preferred file format for github/bitbucket readmes.
If you don't have a program to view this file in its formatted form, head to https://markdownlivepreview.com/ and paste this text on there.
-->

# IWI-Miner
Implementation of an IWI-Miner in cpp. Alongside Python code to cluster stock data into sectors and backtest mining output.

## Notes
------

C++ binary is compiled for linux x64

## Requirements
-----
Python 3 with the packages in requirements.txt installed.
This guide is made for Linux, there is nothing stopping you from running this in windows. But be advised that you must compile your own C++ binary, using the C++ libaries: boost and ltbb.

## Setup
-----
First, make sure you have [python3](https://www.python.org/downloads/) installed. In debian based linux distros, you can install it via apt:
```console
sudo apt-get install python3
```
Next, you must install all the required python packages. You can use a python [virtual environment](https://virtualenv.pypa.io/en/latest) if you are familiarized with this tool, and/or you do not wish to clutter your global pip install.
```console
pip install -r requirements.txt
```
## Configuration
-----
The JSON file config.json .

contains every configurable value.
The default config file contains:

```json
{
    "dateStart": 
    {
        "year" : 2012,
        "month" : 8,
        "day"   : 1,

        "description" : "Starting date for the training period."
    },
    "dateEnd" : 
    {
        "year" : 2013,
        "month" : 8,
        "day"   : 1,
        "description" : "Ending date for the training period." 
    },
    "dataSet" : "nasdaq",
    "minSup" : "8",
    "minDiv" : "70",
    "granularity" : "static"
}
```
Values for 'dataSet' can be: 'nasdaq', 'spy', or 'dow'.\
minSup is the minimum support, i.e. minimum return of the portfolio.\
minDiv is the minimum diversification, i.e. number of unique sectors divided by the number of sectors present.


granularity is used for rolling window mining, it defines the size of the time window. For example, a value of "month" would produce run the mining algorithm every month starting from "dateStart" up to "dateEnd". Valid values for granularity are: 
* 'month': every month
* 'trimester': every three months
* 'year': every year
* 'static': Not rolling window. Training period is the whole time between 'dateStart' and 'dateEnd'.

## Running
---
To run the miner, just perform
```console
python main.py
```
In case apt played funny and installed python under the alias of python3, do:
```console
python3 main.py
```
Either one should work, if your instalation of python and the requirements was performed correctly.

## Recomendations
---
The rolling window feature (any 'granularity' value other than 'static') has **not been thoroughly tested**, therefore it is not recommended to use it. The excpetion to this is a granularity of 'month', with a dateStart and dateEnd exactly one year appart (just like in the default config.json). This scenario has been tested and should work.

Linux is highly recomended. The python portion should work fine without linux, but the C++ would need to be re-compiled for another operating system.
