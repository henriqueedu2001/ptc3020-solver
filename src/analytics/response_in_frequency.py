import numpy as np
from typing import *
from analytics.transfer_function import TransferFunction

def response_in_frequency_sample(
    transfer_function: TransferFunction,
    lower_limit:float = 0.1,
    upper_limit:float = 10000,
    n_points:float = 200
) -> List[np.typing.ArrayLike]:
    """Samples a set of frequencies in the specified range and computes the response
    in frequency for the transfer function G(s). The sampled array of frequencies 
    [w1, w2, w3, ..., wn] is in log scale, for the Bode's Diagram. The output is the
    array of complex numbers [G(jw1), G(jw2), G(jw3), ..., G(jwn)].

    Args:
        transfer_function (TransferFunction): _description_
        lower_limit (float, optional): the lowest frequency of the sample. Defaults to 0.1.
        upper_limit (float, optional): the highest frequency of the sample. Defaults to 10000.
        n_points (float, optional): the number of points of the sample. Defaults to 200.

    Returns:
        List[np.typing.ArrayLike]: the sampled frequencies
    
    Examples:
        >>> G = TransferFunction(
        ...     gain=10,
        ...     poles=[-2],
        ...     zeros=[-1],
        ...     print_style='fraction'
        ... )

        >>> response_in_frequency_sample(
        ...     transfer_function=G,
        ...     lower_limit=0.01,
        ...     upper_limit=100,
        ...     n_points=4
        ... )
        [np.complex128(5.000124996875079+0.024999375015624613j), np.complex128(5.057354322463396+0.53243036541105j), np.complex128(9.217047901907785+1.817070857879262j), np.complex128(9.998000799680128+0.09996001599360255j)]
    """
    frequencies = sample_frequencies(
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        n_points=n_points
    )
    complex_points = response_in_frequency(transfer_function=transfer_function, frequencies=frequencies)
    return complex_points


def response_in_frequency(
    transfer_function: TransferFunction,
    frequencies: np.typing.ArrayLike
) -> List[np.typing.ArrayLike]:
    """Computes the response in frequency for the transfer function G(s), by computing
    the quantity G(jw), for every w in the frequencies array [w1, w2, w3, ..., wn].

    Args:
        transfer_function (TransferFunction): the transfer function
        frequencies (np.typing.ArrayLike): the array of frequencies for calculating the 
        response in frequency

    Returns:
        List[np.typing.ArrayLike]: the response in frequency complex points, as an array
        [G(jw1), G(jw2), G(jw3), ..., G(jwn)]
    """
    frequency_args = (1j) * np.array(frequencies)
    complex_points = [transfer_function.compute(input=w) for w in frequency_args]
    return complex_points


def sample_frequencies(lower_limit: float, upper_limit: float, n_points: int) -> List[np.typing.ArrayLike]:
    """Samples frequencies in logscale.

    Args:
        lower_limit (float): the lowest frequency of the sample
        upper_limit (float): the highest frequency of the sample
        n_points (int): the number of points of the sample

    Returns:
        List[np.typing.ArrayLike]: the sampled frequencies
    
    Examples:
    >>> sample_frequencies(lower_limit=0.001, upper_limit=10000, n_points=8)
    [1.e-03 1.e-02 1.e-01 1.e+00 1.e+01 1.e+02 1.e+03 1.e+04]
    
    >>> sample_frequencies(lower_limit=1, upper_limit=10, n_points=10)
    [ 1.          1.29154967  1.66810054  2.15443469  2.7825594   3.59381366 4.64158883  5.9948425   7.74263683 10.        ]
    """
    start, stop = np.log10(lower_limit), np.log10(upper_limit)
    sample_frequencies = np.logspace(start=start, stop=stop, num=n_points, base=10)
    return sample_frequencies