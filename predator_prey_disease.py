print(''' dx/dt = a*x - b*x*y -c*x \n
    dy/dt= -d*y + e*x*y ''')

import numpy as np, matplotlib.pyplot as plt
dt=0.003
tf=20
t=np.arange(0,tf,dt)
time=np.size(t)
a=4
b=2
#c=2 # c<a
d=4
e=3
C=6
Ze=[0 for i in range(C)]
Xe=[0 for i in range(C)]
for c in range(C):
	def f(x,y):
		return a*x - b*x*y -c*x

	def g(x,y):
		return e*x*y - d*y

	def RK(p,q,r,s,z):
		return ( p + dt*( q + 2*( r + s ) + z )/6 )

	x=np.zeros(time)
	y=np.copy(x)
	x[0]=10
	y[0]=5
	for i in range(time-1):
		k1x=f(x[i],y[i])
		k1y=g(x[i],y[i])
		k2x=f(x[i] + (k1x*dt)/2, y[i] + (k1y*dt)/2)
		k2y=g(x[i] + (k1x*dt)/2, y[i] + (k1y*dt)/2)
		k3x=f(x[i] + (k2x*dt)/2, y[i] + (k2y*dt)/2)
		k3y=g(x[i] + (k2x*dt)/2, y[i] + (k2y*dt)/2)
		k4x=f(x[i] + k3x*dt, y[i] + k3y*dt)
		k4y=g(x[i] + k3x*dt, y[i] + k3y*dt)
		x[i+1]=RK(x[i],k1x,k2x,k3x,k4x)
		y[i+1]=RK(y[i],k1y,k2y,k3y,k4y)
	
	Ze[c]=x
	Xe[c]=y
	plt.plot(x,y,label='prey diseased')
	plt.xlabel('Prey population')
	plt.ylabel('Predator population')
	plt.legend()
	plt.show()
	plt.plot(t,x,label='Prey')
	plt.plot(t,y,label='Predator')
	plt.xlabel('time')
	plt.ylabel('Species Population')
	plt.legend()
	plt.show()
#print(Ze,Xe)
Ze=np.array(Ze)
Xe=np.array(Xe)

q=np.shape(Ze)[0]
for i in range(q):
	x1=Ze[i:i+1].reshape(np.shape(Ze)[1],)
	y1=Xe[i:i+1].reshape(np.shape(Xe)[1],)
	#print("x1 is {}:\n".format(x1))
	#print("y1 is {}:\n".format(y1))
	plt.plot(x1,y1,label='c={}'.format(i))
	
plt.xlabel('Prey population')
plt.ylabel('Predator population')
plt.title('Predator vs Prey population (diseased prey)')
plt.legend()
plt.show()

for i in range(q):
	x1=Ze[i:i+1].reshape(np.shape(Ze)[1],)
	y1=Xe[i:i+1].reshape(np.shape(Xe)[1],)
	#print("x1 is {}:\n".format(x1))
	#print("y1 is {}:\n".format(y1))
	plt.plot(t,x1,label='c={}'.format(i))

plt.xlabel('time')
plt.ylabel('Prey population')
plt.title('Time evolution of prey population (diseased prey)')
plt.legend()
plt.show()

for i in range(q):
	x1=Ze[i:i+1].reshape(np.shape(Ze)[1],)
	y1=Xe[i:i+1].reshape(np.shape(Xe)[1],)
	#print("x1 is {}:\n".format(x1))
	#print("y1 is {}:\n".format(y1))
	plt.plot(t,y1,label='c={}'.format(i))

plt.xlabel('time')
plt.ylabel('Predator population')
plt.title('Time evolution of predator population (diseased prey)')
plt.legend()
plt.show()