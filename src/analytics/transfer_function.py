import numpy as np
from typing import *

class TransferFunction:
    def __init__(
        self,
        gain: Union[float, complex] = None,
        zeros: Union[float, complex, List[float], List[complex], np.typing.ArrayLike] = None,
        poles: Union[float, complex, List[float], List[complex], np.typing.ArrayLike] = None,
        numerator_coefficients: np.typing.ArrayLike = None,
        denominator_coefficients: np.typing.ArrayLike = None,
        print_style: Literal['single_line', 'fraction'] = 'fraction'
    ):
        # Transfer Function in the form G(s) = k(s - z1)(s - z2)...(s - zn)/(s - p1)(s - p2)...(s - pn)
        self.gain = gain
        self.zeros = zeros
        self.poles = poles
        
        # Transfer Function in the form G(s) = (an.s^n + ... + a2.s^2 + a1.s + a0)/(bm.s^m + ... + b2.s^2 + b1.s + b0)
        self.numerator_coefficients = numerator_coefficients
        self.denominator_coefficients = denominator_coefficients
        self.explicit_numerator_coefficients = True if self.numerator_coefficients else False
        self.explicit_denominator_coefficients = True if self.denominator_coefficients else False
        
        self.print_style = print_style
        
        if self.explicit_numerator_coefficients:
            self.numerator_coefficients = TransferFunction.cast_polynomial(self.numerator_coefficients)
        else:
            self.gain = TransferFunction.cast_polynomial(self.gain)
            self.zeros = TransferFunction.cast_polynomial(self.zeros)
        if self.explicit_denominator_coefficients:
            self.denominator_coefficients = TransferFunction.cast_polynomial(self.denominator_coefficients)
        else:
            self.poles = TransferFunction.cast_polynomial(self.poles)
    
    
    def cast_polynomial(polynomial: Union[float, complex, List[complex], np.typing.ArrayLike]) -> np.typing.ArrayLike:
        """Casts a given polynomial in to an numpy complex array or a complex number. The input can be expressed as a
        single number, a python list of numbers or a numpy array; every number in the input can have int, float or 
        complex types. The output has always the type <class 'numpy.complex64'>, if the input is a single number, or 
        <class 'numpy.ndarray'>, with dtype=np.complex64, if the input is a list or an array.

        Args:
            polynomial (Union[float, complex, List[complex], np.typing.ArrayLike]): th input polynomial

        Returns:
            np.typing.ArrayLike: the casted polynomial
        
        Examples:
            >>> cast_polynomial(None)
            (1+0j)
            
            >>> cast_polynomial(10)
            (10+0j)
            
            >>> cast_polynomial([2, 5])
            [2.+0.j 5.+0.j]
            
            >>> cast_polynomial([1 + 3j])
            [1.+3.j]
        """
        if polynomial is None: return np.complex64(1)
        elif type(polynomial) in [list, np.ndarray]: return np.array(polynomial, dtype=np.complex64)
        elif type(polynomial) in [int, float]: return np.complex64(polynomial)
        return polynomial
    
    
    def compute(self, input: Union[int, float, complex]) -> Union[int, float, complex]:
        """Computes the transfer function G(s) for a given complex number input s.

        Args:
            input (Union[int, float, complex]): the input s

        Returns:
            Union[int, float, complex]: the output G(s)
        
        Examples:
            >>> TransferFunction(gain=12, zeros=[-5, -2], poles=[-3, -1]).compute(1)
            (40+0j)
            
            >>> TransferFunction(numerator_coefficients=[2, 3], denominator_coefficients=[1, -2]).compute(3)
            (9+0j)
        """
        G_s_numerator = 1 + 0j
        G_s_denominator = 1 + 0j
        G_s = 1 + 0j
        s = input
        
        # computing the numerator
        if self.explicit_numerator_coefficients:
            # explicit polynomial coefficients
            cum_sum = 0 + 0j
            for i, coeff in enumerate(self.numerator_coefficients):
                term_exp = len(self.numerator_coefficients) - i - 1
                cum_sum += coeff * (s ** term_exp)
            G_s_numerator = cum_sum
        else:
            # implicit polynomial coefficients (just roots)
            G_s_numerator = self.gain * np.prod(s - self.zeros) if self.zeros is not None else 1 + 0j
        
        # computing denominator
        if self.explicit_denominator_coefficients:
            # explicit polynomial coefficients
            cum_sum = 0 + 0j
            for i, coeff in enumerate(self.denominator_coefficients):
                term_exp = len(self.denominator_coefficients) - i - 1
                cum_sum += coeff * (s ** term_exp)
            G_s_denominator = cum_sum
        else:
            # implicit polynomial coefficients (just roots)
            G_s_denominator = np.prod(s - self.poles) if self.poles is not None else 1 + 0j
        
        G_s = G_s_numerator / G_s_denominator
        
        return G_s
    
    
    def __str__(self):
        transfer_function_str = ''
        
        # numerator str
        numerator_str = ''
        if not self.explicit_numerator_coefficients:
            if self.gain.real != 1: numerator_str += str(self.gain.real)
            numerator_str += TransferFunction.polynomial_with_roots_str(self.zeros)
        else:
            numerator_str += TransferFunction.polynomial_with_coefficients_str(self.numerator_coefficients)
        
        # denominator str
        denominator_str = ''
        if not self.explicit_denominator_coefficients:
            denominator_str += TransferFunction.polynomial_with_roots_str(self.poles)
        else:
            denominator_str += TransferFunction.polynomial_with_coefficients_str(self.denominator_coefficients)    
        
        # printing styles
        if self.print_style == 'single_line':
            # single line style printing 
            transfer_function_str = numerator_str + '/' + denominator_str
        elif self.print_style == 'fraction':
            # fraction style printing
            fraction_horizontal_length = max(len(numerator_str), len(denominator_str))
            numerator_left_padding = ' ' * ((fraction_horizontal_length - len(numerator_str)) // 2)
            denominator_left_padding = ' ' * ((fraction_horizontal_length - len(denominator_str)) // 2)
            transfer_function_str += numerator_left_padding + numerator_str
            transfer_function_str += f'\n{fraction_horizontal_length * '-'}\n'
            transfer_function_str += denominator_left_padding + denominator_str + '\n'
        else:
            # invalid printing style
            transfer_function_str = 'invalid printing style'
        
        return transfer_function_str
    
    
    def polynomial_with_roots_str(roots: np.typing.ArrayLike) -> str:
        """Generates the string that represents the complex polynomial, when it's expressed 
        by a list of their roots. The print style is '(s - r1)(s - r2)(s - r3)...(s - rn)',
        when the input polynomial have the roots [r1, r2, r3, ..., rn].

        Args:
            roots (np.typing.ArrayLike): the roots of the polynomial

        Returns:
            str: the polynomial string
            
        Examples:
            >>> TransferFunction.polynomial_with_roots_str([-3, 2, 7])
            (s + 3)(s - 2)(s - 7)
            
            >>> TransferFunction.polynomial_with_roots_str([1 + 2j, 1 - 2j])
            (s - 1.0 - 2.0j)(s - 1.0 + 2.0j)
            
            >>> TransferFunction.polynomial_with_roots_str([-1 + 5j, -1 + 7j])
            (s + 1.0 - 5.0j)(s + 1.0 - 7.0j)
        """
        polynomial_str = ''
        
        # case when the roots are a single complex number
        if type(roots) in [complex, float, int]:
            if roots.imag != 0: return f'({roots})'
            else: return f'({roots.real})'
        
        # case when the roots are a list of complex numbers
        for root in roots:
            real_signal = '-' if root.real > 0 else '+'
            if root.imag != 0:
                imag_signal = '-' if root.imag > 0 else '+'
                polynomial_str += f'(s {real_signal} {abs(root.real)} {imag_signal} {abs(root.imag)}j)'
            else:
                polynomial_str += f'(s {real_signal} {abs(root.real)})'
        return polynomial_str
    
    
    def polynomial_with_coefficients_str(coefficients: np.typing.ArrayLike) -> str:
        """Generates the string that represents the complex polynomial, when it's expressed 
        by a list of their coefficients. The print style is 'an.s^n + ... + a2.s^2 + a1.s + a0',
        when the input polynomial have the coefficients [an, ..., a2, a1, a0].

        Args:
            coefficients (np.typing.ArrayLike): the coefficients of the polynomial

        Returns:
            str: the polynomial string
        
        Examples
            >>> TransferFunction.polynomial_with_coefficients_str([1, 2, 3])
            1s^2 + 2s^1 + 3
            
            >>> TransferFunction.polynomial_with_coefficients_str(10)
            10
            
            >>> TransferFunction.polynomial_with_coefficients_str([7])
            7
            
            >>> TransferFunction.polynomial_with_coefficients_str(np.array([1.0 + 3j, 2 + 5j]))
            (1+3j)s^1 (2+5j)
            
            >>> TransferFunction.polynomial_with_coefficients_str([-1.0 + 3j, -5, -4 - 2j, 12])
            (-1+3j)s^3 - 5s^2 + (-4-2j)s^1 + 12
        """
        polynomial_str = ''
        
        if type(coefficients) in [float, int]: return coefficients
        elif type(coefficients) == complex:
            if coefficients.imag != 0: return str(coefficients)
            else: return str(coefficients.real)
            
        polynomial_degree = len(coefficients) - 1
        for i, coeff in enumerate(coefficients):
            s_term_exp = len(coefficients) - i - 1
            s_term_str = '' if s_term_exp == 0 else f's^{s_term_exp}'
            if coeff.imag != 0:
                optional_signal = '+ ' if s_term_exp != polynomial_degree else ''
                polynomial_str += f'{optional_signal}{coeff}{s_term_str} '
            else:
                real_signal = '' if s_term_exp == polynomial_degree else '+' if coeff.real > 0 else '-'
                optional_space = ' ' if s_term_exp != polynomial_degree else ''
                polynomial_str += f'{real_signal}{optional_space}{abs(coeff.real)}{s_term_str} '
        
        return polynomial_str
