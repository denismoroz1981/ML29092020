##  Perceptron_V

I have refractored file *061_perceptron.py* containing perceptron model with visualization of result.

Classes *Accumulator* and *Animator* have been moved out to separate files as might be used in other projects.

Perceptron model has been moved to separate file as well. Perceptron model has become a class with name 
*"Perceptron_V"*. When creating an object of this class, number of neurons in layers should be indicated as parameters
to set layout of neuron network.

```
perceptron = Perceptron_V(num_inputs,num_hiddens,num_outputs)
```     

For perceptron fulfills its functions two user methods has been organized - *"train"* and *"predict"*. For training tuning
we may indicate loss function ("cross-enthropy" only so far) and activation function: "identical", "sigmoid", "relu" or
"lrelu" (for leaky relu).

```
perceptron.train(train_iter, test_iter, loss, af, batch_size, num_epochs, lr)
```

For prediction we need to pass test sample to respectine method.

```
perceptron.predict(test_iter)
``` 

For demonstration a perceptron object is created in file *main.py* where function for mnist data load is located as has
no direct relation to the perceptron class.
 
