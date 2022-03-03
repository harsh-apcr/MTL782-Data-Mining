"""
General Instructions
Please follow these general instructions so that it'll be easy for others to use your code without contacting you

* Mark all your private functions by the convention - def _private_function()

* You can use the special variable __all__ to define exactly which functions to expose and which not
  when the other users import from your module

* If you need more than one files to complete your assignment (for modularity and organisation purpose), create a new
  python package inside the src folder instead flooding this folder with multiple modules

* Document your codes well, use doc-strings to document instead of single line comments
  before definition of a function, provide the following information
  Summary: Explain in a simple language what the function suppose to do
  Parameters: Explain what are the parameters and parameter-types which the function expects
  Return value: Explain what the function returns

An example is given below
"""


__all__ = ['sum_squares', 'imp_constant']

imp_constant = 100
""" Summary :
       Calculates the sum of square of x and y.     
    Parameters:     
       x:int , a number
       y:int , another number    
    Returns:     
       int: sum of square of x and y
    """


def sum_squares(x, y):
    return _private_square(x)+_private_square(y)


def _private_square(x):
    return x**2
