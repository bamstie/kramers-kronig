# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:41:01 2024

@author: bamer
"""

import numpy
import scipy
import astropy

def _KKR_real2imag_hilbert(
        Eps :numpy.ndarray
        ):
    #pre factor must corrected
    KKR = scipy.signal.hilbert(numpy.real(Eps))
    KKR = -2 / numpy.pi * numpy.imag(KKR)
    return(KKR)

def _KKR_imag2real_hilbert(
        Eps :numpy.ndarray
        ):
    #pre factor must corrected
    KKR = scipy.signal.hilbert(numpy.imag(Eps))
    KKR = 1 + 2 / numpy.pi * numpy.imag(KKR) 
    return(KKR)


def KKR_real2imag(
        Eps_real :numpy.ndarray,
        Energy :astropy.units.core.Unit
        ):
    if (numpy.shape(Eps_real) != numpy.shape(Energy)):
        raise ValueError("energy and eps needs same shape" + 
                         "you entered:" + str(numpy.shape(Eps_real))
                         + str(numpy.shape(Energy))
                         )
    # unit conver here 
    Energy = Energy.value
    
    # Extrapolate a spline to estimate values outside the original range
    expol_energy = numpy.linspace(min(Energy)*0.75, 
                                         max(Energy)*1.25,
                                         len(Energy) * 2)  # Extend the range
    Eps_real_expol = extrapolate_data(Energy, Eps_real, expol_energy)
    # take empirical varianz as error values
    std_Eps_real =  numpy.array([(x - numpy.mean(Eps_real_expol))
                                 /numpy.sqrt(len(Eps_real_expol))
                    for x in Eps_real_expol
                    ])
                                
    std_Energy  =  numpy.array([(x - numpy.mean(expol_energy)) 
                                / numpy.sqrt(len(expol_energy))
                    for x in expol_energy
                    ])
    
    # Calculate the imaginary part using Kramers-Kronig relation
    Eps_imag_expol = numpy.zeros_like(Eps_real_expol, dtype=numpy.complex128)
    std_Eps_imag_expol = numpy.zeros_like(Eps_real_expol, dtype=numpy.complex128)
    
    for i in range(len(expol_energy)):
        integrand = numpy.divide(Eps_real_expol - 1,
                                 expol_energy - expol_energy[i])
        integrand[i] = 0  # Avoid division by zero by setting the problematic term to zero
        integral = numpy.trapz(integrand, expol_energy)
        Eps_imag_expol[i] = -(1 / numpy.pi) * integral
        
        # Calculate the standard deviation of eps_imag based on error propagation
        denominator = expol_energy - expol_energy[i]
        denominator[i] = 1  # Avoid division by zero by setting the problematic term to non-zero
        term1 = numpy.square(Eps_imag_expol[i] / (denominator)**2* std_Eps_real[i])
        term2 = numpy.square(Eps_imag_expol[i] / (denominator)**2* std_Energy[i])
        std_Eps_imag_expol[i] = numpy.sqrt(numpy.sum(term1) + numpy.sum(term2))
    # interpolatate to orginal energy spacing 
    Eps_imag = interpolate_data(expol_energy, Eps_imag_expol, Energy)
    std_Eps_imag = interpolate_data(expol_energy, std_Eps_imag_expol, Energy)

    return(Eps_imag, std_Eps_imag)

def KKR_imag2real(
        Eps_imag :numpy.ndarray,
        Energy :astropy.units.core.Unit
        ):
    if (numpy.shape(Eps_imag) != numpy.shape(Energy)):
        raise ValueError("energy and eps needs same shape" + 
                         "you entered:" + str(numpy.shape(Eps_imag))
                         + str(numpy.shape(Energy))
                         )
    # unit conver here 
    Energy = Energy.value

    # Extrapolate a spline to estimate values outside the original range
    expol_energy = numpy.linspace(min(Energy)*0.75, 
                                         max(Energy)*1.25,
                                         len(Energy) * 2)  # Extend the range
    Eps_imag_expol = extrapolate_data(Energy, Eps_imag, expol_energy)
    # take empirical varianz as error values
    std_Eps_imag = numpy.array([(x - numpy.mean(Eps_imag_expol))
                                /numpy.sqrt(len(Eps_imag_expol))
                    for x in Eps_imag_expol
                    ])
    std_Energy  =  numpy.array([(x - numpy.mean(expol_energy)) 
                                / numpy.sqrt(len(expol_energy))
                    for x in expol_energy
                    ])
    
    # Calculate the imaginary part using Kramers-Kronig relation
    Eps_real_expol  = numpy.zeros_like(Eps_imag_expol, dtype=numpy.complex128)
    std_Eps_real_expol = numpy.zeros_like(Eps_imag_expol, dtype=numpy.complex128)
    
    for i in range(len(expol_energy)):
        integrand = numpy.divide(Eps_imag_expol,
                                 expol_energy - expol_energy[i])
        integrand[i] = 0  # Avoid division by zero by setting the problematic term to zero
        integral = numpy.trapz(integrand, expol_energy)
        Eps_real_expol[i] = 1 + (1 / numpy.pi) * integral
        
        # Calculate the standard deviation of eps_imag based on error propagation
        denominator = expol_energy - expol_energy[i]
        denominator[i] = 1  # Avoid division by zero by setting the problematic term to non-zero
        term1 = numpy.square(Eps_real_expol[i] / (denominator)**2* std_Eps_imag[i])
        term2 = numpy.square(Eps_real_expol[i] / (denominator)**2* std_Energy[i])
        std_Eps_real_expol[i] = numpy.sqrt(numpy.sum(term1) + numpy.sum(term2))
        
    # interpolatate to orginal energy spacing 
    Eps_real = interpolate_data(expol_energy, Eps_real_expol, Energy)
    std_Eps_real = interpolate_data(expol_energy, std_Eps_real_expol, Energy)
    return(Eps_real, std_Eps_real)

def extrapolate_data(
                    X :numpy.ndarray,
                    Y :numpy.ndarray,
                    New_X :numpy.ndarray,
                    ):
    # Extrapolate a spline to estimate values outside the original range
    flip = False 
    # flip if X ins not striclty increasing
    if X[0] > X[len(X)-1]: #falling
        flip = True
        X = numpy.flip(X)
        Y = numpy.flip(Y)
        New_X = numpy.flip(New_X)
    spline = scipy.interpolate,UnivariateSpline(X, Y, k=1, s=0)
    New_Y = spline(New_X)
    if flip == True:
        New_Y = numpy.flip(New_Y)
    return(New_Y)

def interpolate_data(
        X :numpy.ndarray,
        Y :numpy.ndarray,
        New_X :numpy.ndarray,
        ):
    flip = False 
    # flip if X ins not striclty increasing
    if X[0] > X[len(X)-1]: #falling
        flip = True
        X = numpy.flip(X)
        Y = numpy.flip(Y)
        New_X = numpy.flip(New_X)
    # Interpolate data using a spline
    spline = scipy.interpolate,UnivariateSpline(X, Y, k=1, s=0)
    New_Y = spline(New_X)
    if flip == True:
        New_Y = numpy.flip(New_Y)
    return(New_Y)
