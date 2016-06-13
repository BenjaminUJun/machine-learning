""" Active-Set Method for Convex Quadratic Programming
Algorithm minimizes a quadratic function of the form:
	Objective:
		min q(x) = (1/2) x.T.dot(G).dot(x) + x.T.c
	Subject to constraints:
		Ax = b
		Ax >= b
	Method: Active set search

Pseudo-code:
	Compute initial feasible solution x_k (k = 0)
		In the case of portfolio optimization, set x_k = (1/n)
		Else: Use LP
	Set W_k as active working set on x_k: List of index on constraints
		Set A_Active according W_k reference
	Set nW_k as inactive working set on x_k
		Set A_inactive according to nW_k reference
	Repeat:
		iteration += 1
		Solve EQP using KKT matrix with Conjugate Gradient methods: 
			Objective:
				min (1/2) p.T.dot(G).dot(p) + g_k.dot(p)
			Subject to constraints:
				a_i.T.dot(p) = 0
				i belongs to  W_k
			Set p_k as solution to EQP.
		if p_k = 0:
			Solve linear system for lambdas:
				A_Active.dot(lambda) = g = G.dot(x_k) + c
			if all(lambda) >= 0:
				return x_optimal = x_k
				break
			else:
				Delete a constraint from the working set
					j = argmin(lambda)
					W_k = W_k \ [j]
		else: # p_k != 0:
			Compute alpha_k over i in nW_k
				alpha_k = min(1, min((b_i - a_i.dot(x_k)) / (a_i.T.dot(p_k)))
				x_k = x_k + alpha_k * p_k
			if alpha_k < 1: # There is a blocking constraint
				W_k.append(argmin(a_k[i])) # constraints i such that alpha < 1 is smallest is a blocking constraint
			else:
				W_k = W_k

General notes:
	G must be positive semidefinite for CG to work well
	Depending on initial conditions and structure of q(x), can be degenerate (cf. Simplex)
"""

import numpy as np

print """Full system with inequality constraints"""
#G = np.array([[2.,0.],[0.,2.]])
#c = np.array([[-2.],[-5.]])
#A = np.array([[1.,-2.],[-1.,-2.],[-1.,2.],[1.,0.],[0.,1.]])
#b = np.array([[-2.],[-6.],[-2.],[0.],[0.]])
G = np.array([[0.0100,0.0018,0.0011],[0.0018,0.0109,0.0026],[0.0011,0.0026,0.0199]])
c = np.array([[-0.0427],[-0.0015],[-0.0285]])
A = np.array([[1.,1.,1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
b = np.array([[1.],[0.],[0.],[0.]])

KKT_Full = np.concatenate((G,A.T),axis = 1)
zeros = np.zeros((A.shape[0],A.shape[0]))
lower = np.concatenate((A,zeros), axis=1)
KKT_Full = np.concatenate((KKT_Full,lower),axis=0)
RHS_Full = np.concatenate((-c,b),axis=0)

print "KKT Full\n",KKT_Full
print "RHS Full\n",RHS_Full

print """Compute feasible initial x_0"""
x_k = np.array([[1./3.],[1./3.],[1./3.]])
print "x_0\n",x_k

print """Set W_k as active working set on x_k: List of index on constraints
Set A_Active according W_k reference"""
W_k = [1]
if len(W_k)==0:
	A_Active =[]
	print "W_k", W_k
else:
	A_Active = [A[W_k[0]-1]]
	for i in range(1,len(W_k)):
		A_Active = np.concatenate((A_Active,[A[W_k[i]-1]]),axis=0)
	print "W_k", W_k
	A_Active = np.array(A_Active)
	print "A_Active\n", A_Active

print """Set nW_k as inactive working set on x_k
Set A_inactive according to nW_k reference """
List = [(i+1) for i in range(len(A))]
nW_k = [x for x in List if x not in W_k]
A_inactive = [A[nW_k[0]-1]]
for i in range(1,len(nW_k)):
	A_inactive = np.concatenate((A_inactive,[A[nW_k[i]-1]]),axis=0)
A_inactive = np.array(A_inactive)
b_inactive = [b[nW_k[0]-1]]
for i in range(1,len(nW_k)):
	b_inactive = np.concatenate((b_inactive,[b[nW_k[i]-1]]),axis=0)
b_inactive = np.array(b_inactive)
print "nW_k", nW_k
print "A_inactive\n", A_inactive
print "b_inactive\n", b_inactive

print "Initial Objective:", (0.5 * x_k.T.dot(G).dot(x_k) + c.T.dot(x_k))[0,0]

print """--- Initiate main loop ---"""
iteration = 0
while True:
	iteration += 1
	print """Solve for p using QP"""
	print """Reset A to active set constraints"""
	if len(W_k) == 0:
		g = G.dot(x_k) + c
		KKT = G
		RHS = -g
	else:
		KKT = np.concatenate((G,A_Active.T),axis = 1)
		zeros = np.zeros((A_Active.shape[0],A_Active.shape[0]))
		lower = np.concatenate((A_Active,zeros), axis=1)
		KKT = np.concatenate((KKT,lower),axis=0)
		b_Active = np.zeros((len(W_k),1))
		g = G.dot(x_k) + c
		RHS = np.concatenate((-g,b_Active),axis=0)

	print "KKT\n",KKT
	print "RHS\n", RHS

	print """Solve KKT for p
Initialize CG solver"""

	p = np.zeros((KKT.shape[0],1))
	r = RHS - KKT.dot(p)
	v = r
	k = 0
	beta = 0.
	epsilon = 1e-20

	while True:
		v = r + beta * v
		alpha = (r.T.dot(r) / v.T.dot(KKT).dot(v))[0,0]
		p = p + alpha * v
		r_old = r
		r = r - alpha * KKT.dot(v)
		if (r.T.dot(r) < epsilon):
			break
		beta = (r.T.dot(r) / r_old.T.dot(r_old))[0,0]
		k += 1
		#print "Iteration",k,"\n", p

	print "p\n",p
	r = KKT.dot(p) - RHS
	print "Residual score:", (r.T.dot(r))[0,0]

	if len(W_k) == 0:
		p_optimal = p
	else:
		p_optimal = p[:len(A_Active[0])]
		p_optimal[abs(p_optimal)<1e-10] = 0 # zero elements below threshold epsilon
		print "p_optimal\n",p_optimal

	print """Determine if p_optimal is zero. If so, compute lambdas. Else, compute alphas
Filter through constraints that are A_inactive.dot(p_optimal) < 0"""

	if all(p_optimal == 0):
		print "G\n", G
		print "x_k\n",x_k
		print "c\n",c
		g = G.dot(x_k) + c
		print "g\n",g
		print "A_Active.T\n", A_Active.T
		if A_Active.T.shape[1] != A_Active.T.shape[0]:
			A_Active_T = A_Active.T
			A_Trunc = A_Active_T[~np.all(A_Active_T == 0, axis = 1)]
			print "A_Trunc\n", A_Trunc
			g_Trunc = g[abs(g)>1e-10]
			g_Trunc = g_Trunc.reshape(len(g_Trunc),1)
			print "g_Trunc\n",g_Trunc
			lambdas = np.linalg.solve(A_Trunc,g_Trunc)
			print "lambdas\n", lambdas
		else:	
			lambdas = np.linalg.solve(A_Active.T,g)
			print "lambdas\n", lambdas

		if all(lambdas>0):
			x_optimal = x_k	
			print "x_optimal\n", x_optimal
			Z = 0.5 * x_k.T.dot(G).dot(x_k) + x_k.T.dot(c)
			print "Z:", Z[0,0]
			Z_return = (-c).T.dot(x_k)
			#print "Return:", Z_return[0,0]
			Var = 0.5 * x_k.T.dot(G).dot(x_k)
			#print "Risk:", Var[0,0]
			Sharpe = Z_return / Var
			#print "Sharpe Ratio:", Sharpe[0,0]
			print "Iterations:", iteration
			print "nW_k",nW_k
			print "W_k",W_k
			break

		else: # Remove from W_k smallest lambda
			lambdas = lambdas.tolist()
			print "min(lambdas)",min(lambdas)
			lambdas_index = lambdas.index(min(lambdas))
			print "lambdas_index", lambdas_index
			print "shape",A_Active.T.shape[1]
			if A_Active.T.shape[1] == 1:
				nW_k.append(W_k[0])
				W_k =[]
			else:
				nW_k.append(W_k[lambdas_index])
				del W_k[lambdas_index]
			print "W_k",W_k
			print "nW_k", nW_k
			if len(W_k)==0:
				A_Active =[]
			else:
				A_Active = [A[W_k[0]-1]]
				for i in range(1,len(W_k)):
					A_Active = np.concatenate((A_Active,[A[W_k[i]-1]]),axis=0)
				A_Active = np.array(A_Active)
				print "A_Active\n", A_Active

	else:
		print "A_inactive.dot(x_k)\n",A_inactive.dot(x_k)
		alpha_lower = A_inactive.dot(p_optimal)
		alpha_index = [i for i,x in enumerate(alpha_lower) if x < 0]
		print "alpha_index",alpha_index
		#alpha_lower = alpha_lower[alpha_lower<0]
		print "alpha_lower\n", alpha_lower
		#alpha_upper = b_Active[0] - A_inactive.dot(x_k)
		print "b_inactive\n", b_inactive
		alpha_upper = b_inactive - A_inactive.dot(x_k)
		print "alpha_upper\n", alpha_upper
		alpha_ratio = alpha_upper / alpha_lower
		alpha_ratio = [alpha_ratio[i] for i in alpha_index]
		alpha_ratio = np.array(alpha_ratio)
		print "alpha_ratio\n",alpha_ratio
		if len(alpha_ratio) == 0:
			alpha_k = 1.
		else:
			alpha_k = min(1.,max(0.,min(alpha_ratio)))

		print "alpha_k", alpha_k
		x_k = x_k + alpha_k * p_optimal
		print "x_k\n",x_k
		Z = 0.5 * x_k.T.dot(G).dot(x_k) + x_k.T.dot(c)
		print "Iteration", iteration,"Objective:",Z[0,0]

		print "If there are block constraints (i.e. alpha <1), update W_k with blocking constraints. Remove from nW_k"
		if alpha_k < 1.:
			#block_index = [i for i,x in enumerate(A_inactive.dot(p_optimal)) if x == alpha_lower][0]
			block_index = alpha_index[0]
			print "block_index",block_index
			print "W_k",W_k
			print "nW_k",nW_k
			W_k.append(nW_k[block_index])
			print "W_k",W_k
			del nW_k[block_index]
			print "nW_k", nW_k

		A_Active = [A[W_k[0]-1]]
		for i in range(1,len(W_k)):
			A_Active = np.concatenate((A_Active,[A[W_k[i]-1]]),axis=0)
		A_Active = np.array(A_Active)
		A_inactive = [A[nW_k[0]-1]]
		b_inactive = [b[nW_k[0]-1]]
		for i in range(1,len(nW_k)):
			A_inactive = np.concatenate((A_inactive,[A[nW_k[i]-1]]),axis=0)
		for i in range(1,len(nW_k)):
			b_inactive = np.concatenate((b_inactive,[b[nW_k[i]-1]]),axis=0)
		A_inactive = np.array(A_inactive)
		b_inactive = np.array(b_inactive)
		print "A_Active\n", A_Active
		print "A_inactive\n", A_inactive
