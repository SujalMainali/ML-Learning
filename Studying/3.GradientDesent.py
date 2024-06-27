# The mean sqare error  can also be called as cost functions


# Gradient decentis an algorithm that finds best fit line for given training data se
# the gradient descent works as follows the cost function is plotted against slope(m) and intersept(b) to make a plane
# The values of m and b are changed randomly untill the lowest cost function is found


# If we are trying to find the m and b for minimum cost function manually. This is what we are suppposed to do
# Start with certain point and create a plot of cost function and m or b. then the different value of m an db is taken and different values of cost function is plotted
# THe different values of m and b can be taken in certain interval the smaller the better to reach a certain lowestt value


# to make a computer do this process we can find the partial derivatives of cost function w.r.t m or b. the partial derivatives gives the direction(change in value of m and b) that we need to move
# then the new value of m is m=m-learning rate * partial derivative of cf w.r.t m

import numpy as np

def gradient_descent(x,y):
    m_curr=0
    b_curr=1000
    iterations=100000
    n=len(x)
    learning_rate= 0.08

    for i in range(iterations):
        y_predicted=m_curr*x + b_curr
        cost= (1/n)*sum([val**2 for val in (y-y_predicted)])
        md=-(2/n)*sum([val for val in x*(y-y_predicted)])
        bd=-(2/n)*sum(val for val in (y-y_predicted))
        m_curr= m_curr- learning_rate*md
        b_curr= b_curr -learning_rate*bd
        print(f"m={m_curr} b={b_curr} cost={cost} iteration={i}")
        # print("MD=")
        # print(md)
        # print("BD=")
        # print(bd)
        # print("M=")
        # print(m_curr)


x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_descent(x,y)