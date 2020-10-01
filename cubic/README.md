##CUBIC.py

Application **cubic.py** allows to find real and complex roots of cubic equation having the following format:
**a**x^3+**b**x^2+**c**x+**d**=0

First, you need to prepare a json-file with paramenters: a,b,c,d.
The structure of the object in the json-file should be like this:
```
{ "a":1, "b":2, "c":3, "d":4} 
```

After entering parameters launch **cubic.py** in Terminal window with compulsory argument - path to your json-file.
```
python cubic.py --path params.json
```
Calculated roots in case of success or information about errors might be found at log-file **app_cubic.log**.

For the code of the application refer to [GitHub page](https://github.com/denismoroz1981/ML29092020/tree/master/cubic).

 

