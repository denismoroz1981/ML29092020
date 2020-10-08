##f_activation.py

**f_activation.py** is an application for calcutation scalar of vector derivative for the following activation functions:

* Logistic function (or sigmoid) <br>
* SoftPlus function <br>
* ArcTan function <br><br>
![image info](./figures/f_act_fig.png)


First, you need to prepare a json-file with paramenters: a scalar and a vector. See example below:<br>
<i>&nbsp;&nbsp;  {&nbsp; "scalar": [1],<br>
&nbsp;&nbsp;    "vector":&nbsp;[[3,-2,-1]]&nbsp; } </i>

After that launch **f_activation.py** in Terminal window with compulsory argument -- path to your json-file.
```
python system.py --path params.json
```
Calculated derivatives or information about errors might be found at the log-file **app_activation.log**.

