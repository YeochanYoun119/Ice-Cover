# Ice-Cover
This program retrieves dates and durations of full-freeze ice covers of Mendota lake in Wisconsin from [Wisconsin State Climatology Office](http://www.aos.wisc.edu/~sco/lakes/Mendota-ice.html). Based on this data, linear regression is implemented to calculate and predict the ice-cover duration for the future years of the Mendota lake.

# Equation
## Linear Regression
This program performs linear regression with the model\
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)=\beta&space;_{0}&plus;\beta&space;_{1}x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)=\beta&space;_{0}&plus;\beta&space;_{1}x" title="f(x)=\beta _{0}+\beta _{1}x" /></a>

## regression(beta_0, beta_1):
This function calculates the mean squared error of betas.\
<a href="https://www.codecogs.com/eqnedit.php?latex=MSE(\beta_{0},&space;\beta_{1})=\frac{1}{n}\Sigma&space;_{i=1}^{n}\textrm(\beta_{0}&plus;\beta_{1}x_{i}-y_{i})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MSE(\beta_{0},&space;\beta_{1})=\frac{1}{n}\Sigma&space;_{i=1}^{n}\textrm(\beta_{0}&plus;\beta_{1}x_{i}-y_{i})^2" title="MSE(\beta_{0}, \beta_{1})=\frac{1}{n}\Sigma _{i=1}^{n}\textrm(\beta_{0}+\beta_{1}x_{i}-y_{i})^2" /></a>\
Two betas represents the two arguments of the function.\
**Returns** corresponding MSE
```
>>> regression(0,0)
=> 10827.78
>>> regression(100,0)
=> 386.57
>>> regression(300,-.1)
=> 332.83
>>> regression(400,.1)
=> 242059.01
>>> regression(200,-.2)
=> 84167.47
```

## gradient_descent(beta_0, beta_1):
This function performs gradient descent on the MSE. At the current parameter(beta_0, beta_1), the gradient is defined by the vector of partial derivatives.\
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\vartheta&space;MSE(\beta_{0},&space;\beta_{1})}{\vartheta\beta_{0}}=\frac{2}{n}\Sigma&space;_{i=1}^{n}\textrm(\beta_{0}&plus;\beta_{1}x_{i}-y_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\vartheta&space;MSE(\beta_{0},&space;\beta_{1})}{\vartheta\beta_{0}}=\frac{2}{n}\Sigma&space;_{i=1}^{n}\textrm(\beta_{0}&plus;\beta_{1}x_{i}-y_{i})" title="\frac{\vartheta MSE(\beta_{0}, \beta_{1})}{\vartheta\beta_{0}}=\frac{2}{n}\Sigma _{i=1}^{n}\textrm(\beta_{0}+\beta_{1}x_{i}-y_{i})" /></a>\
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\vartheta&space;MSE(\beta_{0},&space;\beta_{1})}{\vartheta\beta_{1}}=\frac{2}{n}\Sigma&space;_{i=1}^{n}\textrm(\beta_{0}&plus;\beta_{1}x_{i}-y_{i})x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\vartheta&space;MSE(\beta_{0},&space;\beta_{1})}{\vartheta\beta_{1}}=\frac{2}{n}\Sigma&space;_{i=1}^{n}\textrm(\beta_{0}&plus;\beta_{1}x_{i}-y_{i})x_{i}" title="\frac{\vartheta MSE(\beta_{0}, \beta_{1})}{\vartheta\beta_{1}}=\frac{2}{n}\Sigma _{i=1}^{n}\textrm(\beta_{0}+\beta_{1}x_{i}-y_{i})x_{i}" /></a>\
**retruns** the corresponding gradient as a tuple with the partial derivative with respect to beta_0 as the first value.

```
>>> gradient_descent(0,0)
=> (-204.41, -395063.04)
>>> gradient_descent(100,0)
=> (-4.41, -7663.04)
>>> gradient_descent(300,-.1)
=> (8.19, 16289.42)
>>> gradient_descent(400,.1)
=> (982.99, 1905384.49)
>>> gradient_descent(200,-.2)
=> (-579.21, -1121958.11)
```

## iterate_gradient(T, eta):
This function calculates\
<a href="https://www.codecogs.com/eqnedit.php?latex=\beta_{0}^{(t)}=\beta_{0}^{(t-1)}-\eta&space;\frac{\vartheta&space;MSE(\beta_{0}^{(t-1)},&space;\beta_{1}^{(t-1)})}{\vartheta&space;\beta_{0}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta_{0}^{(t)}=\beta_{0}^{(t-1)}-\eta&space;\frac{\vartheta&space;MSE(\beta_{0}^{(t-1)},&space;\beta_{1}^{(t-1)})}{\vartheta&space;\beta_{0}}" title="\beta_{0}^{(t)}=\beta_{0}^{(t-1)}-\eta \frac{\vartheta MSE(\beta_{0}^{(t-1)}, \beta_{1}^{(t-1)})}{\vartheta \beta_{0}}" /></a>\
<a href="https://www.codecogs.com/eqnedit.php?latex=\beta_{1}^{(t)}=\beta_{1}^{(t-1)}-\eta&space;\frac{\vartheta&space;MSE(\beta_{0}^{(t-1)},&space;\beta_{1}^{(t-1)})}{\vartheta&space;\beta_{1}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta_{1}^{(t)}=\beta_{1}^{(t-1)}-\eta&space;\frac{\vartheta&space;MSE(\beta_{0}^{(t-1)},&space;\beta_{1}^{(t-1)})}{\vartheta&space;\beta_{1}}" title="\beta_{1}^{(t)}=\beta_{1}^{(t-1)}-\eta \frac{\vartheta MSE(\beta_{0}^{(t-1)}, \beta_{1}^{(t-1)})}{\vartheta \beta_{1}}" /></a>\
parameter T is number of iterations to perform, eta is the prameter for the above calculations.\
Always begin from initial parameter (0,0)\
**prints** the followings:
1. the current iteration number beginning at 1 and ending at T
2. the current value of beta_0
3. the current value of beta_1
4. the current MSE
```
>>> iterate_gradient(5, 1e-7)
1 0.00 0.04 1079.72
2 0.00 0.05 474.59
3 0.00 0.05 437.03
4 0.00 0.05 434.69
5 0.00 0.05 434.55
>>> iterate_gradient(5, 1e-8)
1 0.00 0.00 9325.63
2 0.00 0.01 8040.58
3 0.00 0.01 6941.27
4 0.00 0.01 6000.84
5 0.00 0.02 5196.33
```

# Run the program
## predict(year):
This function predicts the days of ice-cover on lake Mendota in given future year.\
**returns** predicted days for future year of frozen Mendota\
You can use this program as a following example: 
```
>>> predict(2050)
=> 80.21
```
