import numpy as np
import matplotlib.pyplot as plt
import cvxopt

# Define model
def line(t,B0,B1,B2):
	return (- B0 - B1 * t)/ B2
n = 1000

# Generate linearly separable data
z_plus = np.ones((n,1))
z_minus = np.ones((n,1))

a = np.random.randn(n,2) + 3# label 1
b = np.random.randn(n,2) # label -1

#plt.plot(a[:,0],a[:,1],'ro',b[:,0],b[:,1],'bo') # plot data
#plt.show()

z_plus = np.concatenate((z_plus,a),axis=1)
z_minus = np.concatenate((z_minus,b),axis=1)

# Classify data
z_plus = np.concatenate((z_plus,np.ones((n,1))),axis=1)
z_minus = np.concatenate((z_minus,-np.ones((n,1))),axis=1)
z_full = np.concatenate((z_minus,z_plus),axis=0)

y = z_full[:,3].tolist()

# Set range for separating plane
x_min_range = min(min(a[:,0]),min(b[:,0]))
x_max_range = max(max(a[:,0]),max(b[:,0]))
t = np.arange(x_min_range,x_max_range)

print "Support Vector Machine"

# Initiate IQP

Q = [[y[i]*y[j]*(z_full[i,1:3].dot(z_full[j,1:3])) for i in range(2*n)] for j in range(2*n)]
Q = cvxopt.matrix(Q)
p = cvxopt.matrix(-np.ones((2*n,1)))

c = 0.1

G = -np.identity(2*n)
G = np.concatenate((G,np.identity(2*n)),axis = 0)
#print G
G = cvxopt.matrix(G)
h = np.zeros((2*n,1))
h = np.concatenate((h,c*np.ones((2*n,1))),axis = 0)
#print h
h = cvxopt.matrix(h)
A = cvxopt.matrix(np.array(y).T,(1,2*n))
B = cvxopt.matrix(0.)
sol=cvxopt.solvers.qp(Q, p, G, h, A, B)

x = np.array(sol['x'])
epsilon = 1e-5
x[x < epsilon] = 0.
alpha = x[x>epsilon]

support_vector_index = np.where(x > epsilon)[0]

sv = z_full[support_vector_index,1:3]
y = np.array(y)
sv_target = np.array([y[support_vector_index]]).T
w = alpha[0]*sv_target[0]*sv[0]
for i in range(1,len(support_vector_index)):
	w +=alpha[i]*sv_target[i]*sv[i]
w = np.array([w]).T

c1 = sv_target[0] - w.T.dot(sv[0])
c2 = sv_target[1] - w.T.dot(sv[1])
c_plus = c1 + 1.
c_minus = c1 - 1.


B0_1 = c1
B0_2 = c2
B1 = w[0]
B2 = w[1]
print "Equation of line: y =", 1./B2,"*(",c1,"-",B1,"* x)"

# Plot
plt.plot(a[:,0],a[:,1],'ro',b[:,0],b[:,1],'bo') # plot data
plt.plot(sv[:,0],sv[:,1],'yo') #plot support vectors
plt.plot(t,line(t,B0_1,B1,B2),'y') # plot separating plane
plt.plot(t,line(t,c_plus,B1,B2),'b') # plot decision boundary
plt.plot(t,line(t,c_minus,B1,B2),'r') # plot decision boundary
plt.title('Soft Margin Linear Support Vector Machine')
plt.show()
