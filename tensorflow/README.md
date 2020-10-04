##System.py

**System.py** is an application for finding solutions of a system of linear equations, having the following format:

**k11** * x1 + **k12** * x2 + **k1i** * xi = **b1** <br>
**k21** * x1 + **k22** * x2 + **k2i** * xi = **b2** <br>
....................................................<br>
**ki1** * x1 + **ki2** * x2 + **kii** * xi = **bi** <br>
<br>
where **k11 - kii** are called "coefficients"and **b1-bi** - "constants". <br>

First, you need to prepare a json-file with paramenters of the system. See example below:<br>
<i>&nbsp;&nbsp;  {&nbsp; "coefficients": [<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    [2,3,1],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    [-2,1,0],<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    [1,2,-2]],<br>
&nbsp;&nbsp;    "constants":&nbsp;[[3,-2,-1]]&nbsp; } </i>

Beware that the number of linear equations should be equal to the number of variables, linear equations should be independent and consistent. In the other cases error about unequal dimensions of the matrixes will be recorded to the log.

After that launch **system.py** in Terminal window with compulsory argument - path to your json-file.
```
python system.py --path params.json
```
The solutions, in case of success, or information about errors might be found at the log-file **app_system.log**.

Solutions are found by inverted matrix method involving multiplication of matrixes: inverted "coefficients" matrix and transposed "constants" matrix. At the log-file you will also find what solutions will be if the order of matrix multiplication is changed: "constants" matrix multiplied by "coefficients" matrix.