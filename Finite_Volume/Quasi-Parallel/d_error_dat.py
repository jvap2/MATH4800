import numpy as np
import matplotlib.pyplot as plt

'''This first section includes the first left derivative numerical experiment'''
C_L_2=np.array([1.0195e-03,2.422e-04,6.0313e-05,1.5026e-05,3.753e-06,9.35e-07,2.777e-07])
C_L_inf=np.array([3.367e-3,8.873e-4,2.229e-4,5.8032e-5,1.4625e-5,3.6711e-6,9.1921e-7])
L_L_2=np.array([4.8e-3,1.426e-3,4.271e-4,1.296e-4,3.9892e-5,1.244e-5,3.9294e-6])
L_L_inf=np.array([5.915e-3,1.775e-3, 5.3673e-4,1.6415e-4,5.0828e-5,1.5922e-5,5.0389e-6])
h=np.array([2**-4,2**-5,2**-6,2**-7,2**-8,2**-9,2**-10])

fig, (ax1,ax2)=plt.subplots(2,1,figsize=(7,14))
ax1.plot(h,C_L_2, label='Cubic')
ax1.plot(h,L_L_2, label='Linear')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('\u0394 h')
ax1.set_ylabel('L\u2082')
ax1.legend()
ax1.set_title('L\u2082 Error')
ax2.plot(h, C_L_inf, label='Cubic')
ax2.plot(h,L_L_inf, label='Linear')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('\u0394 h')
ax2.set_ylabel('L\u221e')
ax2.legend()
ax2.set_title('L\u221e Error')
fig.suptitle('Numerical Experiment 1 Error Data')
plt.savefig("exp1_err.pdf")
plt.show()


C_L_2=np.array([2.232e-03,5.551e-04,1.385e-04,3.4591e-05,8.667e-06,2.1741e-06,6.468e-06])
C_L_inf=np.array([5.711e-3,1.471e-3,3.733e-4,9.405e-5,2.361e-5,5.9289e-6,1.3779e-5])
L_L_2=np.array([1.124e-3,4.639e-4,3.059e-4,1.686e-4,8.45e-5,4.032e-5,1.872e-6])
L_L_inf=np.array([4.108e-3,1.853e-3, 9.196e-4,4.394e-4,2.064e-4,9.575e-5,4.399e-5])
h=np.array([2**-4,2**-5,2**-6,2**-7,2**-8,2**-9,2**-10])

fig, (ax1,ax2)=plt.subplots(2,1,figsize=(7,14))
ax1.plot(h,C_L_2, label='Cubic')
ax1.plot(h,L_L_2, label='Linear')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('\u0394 h')
ax1.set_ylabel('L\u2082')
ax1.legend()
ax1.set_title('L\u2082 Error')
ax2.plot(h, C_L_inf, label='Cubic')
ax2.plot(h,L_L_inf, label='Linear')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('\u0394 h')
ax2.set_ylabel('L\u221e')
ax2.legend()
ax2.set_title('L\u221e Error')
fig.suptitle('Numerical Experiment 1 Error Data')
plt.savefig("exp2_err.pdf")
plt.show()