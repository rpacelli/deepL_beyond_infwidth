from __future__ import print_function
import os, os.path, sys, time, math
import numpy as np
import torch

def k0(x,y,lambda0):
    N0 = len(x)
    return (1/(lambda0*N0)) * np.dot(x,y)

def CorrMat(P,data,lambda0):
    C = np.zeros((P,P))
    for i in range(P): 
        for j in range(i,P):         
            C[i][j] = k0(data[i], data[j], lambda0)
            C[j][i] = C[i][j]
    return C  

def kmatrix(P,C,kernel,lambda1):
    K = np.zeros((P,P))
    for i in range(P): 
        for j in range(i,P):         
            K[i][j] = kernel(C[i][i], C[i][j], C[j][j],lambda1)
            K[j][i] = K[i][j]
    return K

def kernel_erf(k0xx,k0xy,k0yy,lambda1):
    return (2/(lambda1*np.pi))*np.arcsin((2*k0xy)/np.sqrt((1+2*k0xx)*(1+2*k0yy)))

def kappa1(u):
    return (1/(2*np.pi))*(u * (np.pi - np.arccos(u))+ np.sqrt(1-u**2))   

def kernel_relu(k0xx, k0xy, k0yy,lambda1):
    if k0xy == k0xx:
        return np.sqrt(k0xx*k0yy)
    else:
        u = k0xy/np.sqrt(k0xx*k0yy)
        kappa = kappa1(u)
        return np.sqrt(k0xx*k0yy)*kappa

def test_error(data, x,y,labels, lambda1, invK, Qbar,lambda0,kernel,L,device):
    P = len(data)
    k0xx = k0(x,x,lambda0)
    k0yyvec = np.random.randn(P)
    Kmu = np.random.randn(P)
    for i in range(P):
        k0xy = k0(x,data[i],lambda0)
        k0yy = k0(data[i],data[i],lambda0)
        k0yyvec[i] = k0yy
        Kmu[i] = kernel(k0xx,k0xy,k0yy,lambda1)
    k0xx = kernel(k0xx,k0xx,k0xx,lambda1)
    for l in range(L-1):
        for i in range(P):
            k0yy = kernel(k0yyvec[i],k0yyvec[i],k0yyvec[i],lambda1)
            k0yyvec[i] = k0yy
            Kmu[i] = kernel(k0xx,Kmu[i],k0yy,lambda1)
        k0xx = kernel(k0xx,k0xx,k0xx,lambda1)
    K0_invK = np.matmul(Kmu, invK)
    sum1 = -np.dot(K0_invK, labels) + y
    sum2 = -np.dot(K0_invK, Kmu) + k0xx
    return sum1**2 - (Qbar)*sum2/lambda1

def qbar(labels, invK, N1,lambda1):
    P = len(labels)
    alpha1 = P/N1
    yky = np.matmul(np.matmul(np.transpose(labels), invK), labels)
    print(f'\ny K-1 y /P is {yky/P}')
    return ((alpha1-1)-np.sqrt((alpha1-1)**2 + 4*alpha1*yky/(lambda1*P)))/2

def compute_theory(data, labels, test_data, test_labels, N1, lambda1, P,Ptest,lambda0,act,L,device,infwidth):
    data = data.detach().cpu()
    labels = labels.detach().cpu()
    test_data = test_data.detach().cpu()
    test_labels = test_labels.detach().cpu()
    K = CorrMat(P,data,lambda0)
    if act == "erf":
        kernel = kernel_erf
    else:
        kernel = kernel_relu
    for i in range(L):
        K = kmatrix(P,K,kernel,lambda1)
    invK = np.linalg.inv(K)
    #invK = invK.to(device)
    if infwidth:
        Qbar = -1
    else:
        Qbar = qbar(labels, invK, N1, lambda1)
    print(f"\nbar Q is {Qbar}")
    gen_error_pred = 0
    for p in range(Ptest):
        x = np.array(test_data[p])
        y = np.array(test_labels[p])
        gen_error_pred += test_error(data, x, y, labels, lambda1, invK, Qbar,lambda0,kernel,L,device).item()
    gen_error_pred = gen_error_pred/Ptest
    return gen_error_pred, Qbar

