from Pyro4 import expose
from random import randint
import random
import time
class Node:
    def __init__(self,u,v,first):
        self.u=u
        self.v=v
        self.first=first
def F(j,i,t):
    #if(j>10 and j<20 and i>10 and i<20 and t<0.25):
    if(t<0.25 and j%50>15 and j%50<35 and i>5 and i<15):
        return 1
    else:
        return 0
class Solver:
    v=[]
    u=[]
    p=[]
    vn=[]
    un=[]
    pn=[]
    b=[]
    ny=0
    nx=0
    rho=0
    nu=0
    num=0
    dt=0
    dx=0
    dy=0
    stepcount=0
    interactions=0
    first=0
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        print("Inited")
    def solve(self):
        print("Job Started")
        print("Workers %d" % len(self.workers))
        (W,H,nx,ny,rho,nu,dt,nit,u,v) = self.read_input()
        dx = (W+0.1-0.1) / (nx - 1)
        dy = (H+0.1-0.1) / (ny - 1)
        un = []
        kw=len(self.workers)
        step=ny//kw+1
        uw=[]
        vw=[]
        nodes=[]
        first=0
        for i in xrange(ny):
            uw.append(list(u[i]))
            vw.append(list(v[i]))
            if((i+1)%step==0):
                if(len(uw)==0 or len(uw)==1):
                    raise Exception(str(i)+"Wrong size of part init")
                nodes.append(Node(uw,vw,first))
                uw=[]
                uw.append(list(u[i-1]))
                uw.append(list(u[i]))
                vw=[]
                vw.append(list(v[i-1]))
                vw.append(list(v[i]))
                first=i-1
        nodes.append(Node(uw,vw,first))
        c=[]
        for i in xrange(kw):
            if(len(nodes[i].u)<=1):
                raise Exception(str(i)+"Wrong size of part initgetnode")
            c.append(self.workers[i].getnode(nodes[i].u,nodes[i].v,nodes[i].first,rho,nu,dt,dx,dy,i,step))
        Solver.interactions+=1
        self.checkInteractions()

        stepcount = 0
        while stepcount<nit:
            c=[]
            for i in xrange(kw):
                c.append(self.workers[i].computeb())
            c
            Solver.interactions+=1
            self.checkInteractions()
            
            
            for q in range(25):
                pborders=[[[],[]]]
                pbordersreturns=[]
                for i in xrange(kw):
                    pbordersreturns.append((self.workers[i].computep()))
                c=[]
                for i in xrange(kw):
                    pborders.append(pbordersreturns[i].value)
                pborders[0][1]=pborders[1][0]
                pborders.append([pborders[-1][1],[]])
                Solver.interactions+=1
                #self.checkInteractions()
                for i in xrange(kw):
                    c.append(self.workers[i].getp([pborders[i][1],pborders[i+1][0]]))
                Solver.interactions+=1
                #self.checkInteractions()
                c
            
            uborders=[]
            vborders=[]
            ubordersreturns=[]
            vbordersreturns=[]
            for i in xrange(kw):
                ubordersreturns.append((self.workers[i].computeu()))
            for i in xrange(kw):
                vbordersreturns.append((self.workers[i].computev()))
            for i in xrange(kw):
                uborders.append(ubordersreturns[i].value)
                vborders.append(vbordersreturns[i].value)
            Solver.interactions+=2
            self.checkInteractions()


            zerovector=[0 for i in xrange(nx)]
            c=[]
            c.append(self.workers[0].getuv(zerovector,zerovector))
            
            #c.append(self.workers[0].getleftv(zerovector)
            
            for i in xrange(1,kw):
                c.append(self.workers[i-1].getuv(uborders[i][0],vborders[i][0]))
                
            #for i in xrange(1,kw):
            #    c.append(self.workers[i-1].getrightv(vborders[i][0]))
            for i in xrange(1,kw):
                c.append(self.workers[i].getuv(uborders[i-1][1],vborders[i-1][1]))
            #for i in xrange(1,kw):
            #    c.append(self.workers[i].getleftv(vborders[i-1][1]))
            
            c.append(self.workers[-1].getuv(zerovector,zerovector))
            c#.append(self.workers[-1].getrightv(zerovector))
            Solver.interactions+=4
            self.checkInteractions()
            c
            stepcount+=1 
        umapped=[]
        vmapped=[]
        for i in xrange(kw):
            umapped.append(self.workers[i].giveansu())
        for i in xrange(kw):
            vmapped.append(self.workers[i].giveansv())
        
        
        uans=self.myreduce(umapped)
        
        vans=self.myreduce(vmapped)
        #step = int(Solver.n / len(self.workers))
        #if((Solver.n%len(self.workers))>0):
        #    step+=1
        #st=0
        #ii=0
        #x=[0 for i in range(Solver.n)]
        #for t in range(500):
        #    mapped = []
        #    for i in xrange(0, len(self.workers)):
        #        mapped.append(self.workers[i].qqq(Solver.n,Solver.A,x,i*step,i*step+step,st))
        #    xnew=self.myreduce(mapped)
        #    norm=0
        #    for i in range(Solver.n):
        #        norm+=abs(xnew[i]-x[i])
        #    if(norm<-1):
        #        break
        #    else:
        #        x=xnew
        #        st=1
        #        Solver.A=[]
        
        self.write_output(W,H,nx,ny,uans,vans)
    @staticmethod
    @expose
    def getnode(u,v,first,rho,nu,dt,dx,dy,num,step):
        Solver.u=u
        Solver.v=v
        Solver.first=first
        Solver.rho=rho
        Solver.nu=nu
        Solver.dt=dt
        Solver.dx=dx
        Solver.dy=dy
        Solver.num=num
        Solver.step=step
        Solver.ny=len(u)
        Solver.nx=len(u[0])
        if(Solver.ny<=1):
            raise Exception(str(i)+"Wrong size of part getnode")
        for i in xrange(Solver.ny):
            Solver.un.append([0 for j in xrange(Solver.nx)])
        Solver.vn = []
        for i in xrange(Solver.ny):
            Solver.vn.append([0 for j in xrange(Solver.nx)])
        Solver.p = []
        for i in xrange(Solver.ny):
            Solver.p.append([0 for j in xrange(Solver.nx)])
        Solver.pn = []
        for i in xrange(Solver.ny):
            Solver.pn.append([0 for j in xrange(Solver.nx)])
        Solver.b = []
        for i in xrange(Solver.ny):
            Solver.b.append([0 for j in xrange(Solver.nx)])
        Solver.interactions+=1
    @staticmethod
    @expose
    def computeb():
        if(Solver.ny<=1):
            raise Exception(str(ny)+"Wrong size of part computeb")
        Solver.un = []
        for i in xrange(Solver.ny):
            Solver.un.append(list(Solver.u[i]))
        Solver.vn = []
        for i in xrange(Solver.ny):
            Solver.vn.append(list(Solver.v[i]))
        Solver.b = []
        for i in xrange(Solver.ny):
            Solver.b.append([0 for j in xrange(Solver.nx)])
    
        for i in xrange(1,Solver.nx-1):
            for j in xrange(1,Solver.ny-1):
                    Solver.b[j][i] = (Solver.rho * (1 / Solver.dt * ((Solver.u[j][i+1] - Solver.u[j][i-1]) / (2 * Solver.dx) +
                                            (Solver.v[j+1][i] - Solver.v[j-1][i]) / (2 * Solver.dy)) -
                                    ((Solver.u[j][i+1] - Solver.u[j][i-1]) / (2 * Solver.dx))**2 -
                                    2 * ((Solver.u[j+1][i] - Solver.u[j-1][i]) / (2 * Solver.dy) *
                                        (Solver.v[j][i+1] - Solver.v[j][i-1]) / (2 * Solver.dx))-
                                    ((Solver.v[j+1][i] - Solver.v[j-1][i]) / (2 * Solver.dy))**2))
        Solver.interactions+=1
    @staticmethod
    @expose
    def computep():
        if(Solver.ny<=1):
            raise Exception(str(len(Solver.p))+"Wrong size of part computep")
        Solver.pn = []
        for i in xrange(Solver.ny):
            Solver.pn.append(list(Solver.p[i]))
        
        for i in xrange(1,Solver.nx-1):
            for j in xrange(1,Solver.ny-1):
                    Solver.p[j][i] = (((Solver.pn[j][i+1] + Solver.pn[j][i-1]) * Solver.dy**2 +
                                (Solver.pn[j+1][i] + Solver.pn[j-1][i]) * Solver.dx**2) /
                                (2 * (Solver.dx**2 + Solver.dy**2)) -
                                Solver.dx**2 * Solver.dy**2 / (2 * (Solver.dx**2 + Solver.dy**2)) * Solver.b[j][i])
        for i in xrange(Solver.ny):
            Solver.p[i][-1]=Solver.p[i][-2]
            Solver.p[i][0]=Solver.p[i][1]
        ans=[]
        ans.append(list(Solver.p[1]))
        ans.append(list(Solver.p[-2]))
        Solver.interactions+=1
        return ans
    @staticmethod
    @expose
    def getp(pvalue):
        Solver.p[0]=(pvalue[0])
        Solver.p[-1]=(pvalue[-1])
        Solver.interactions+=1
    @staticmethod
    @expose
    def getleftp(pw):
        Solver.p[0]=list(pw)
        Solver.interactions+=1
    @staticmethod
    @expose
    def getrightp(pw):
        Solver.p[-1]=list(pw)
        Solver.interactions+=1
    @staticmethod
    @expose
    def computeu():
        Solver.stepcount+=1
        for i in xrange(1,Solver.nx-1):
            for j in xrange(1,Solver.ny-1):
                Solver.u[j][i] = (Solver.un[j][i] -
                            Solver.un[j][i] * Solver.dt / Solver.dx * 
                            (Solver.un[j][i] - Solver.un[j][i-1]) -
                            Solver.vn[j][i] * Solver.dt / Solver.dy * 
                            (Solver.un[j][i] - Solver.un[j-1][i]) -
                            Solver.dt / (2 * Solver.rho * Solver.dx) * 
                            (Solver.p[j][i+1] - Solver.p[j][i-1]) +
                            Solver.nu * (Solver.dt / Solver.dx**2 * 
                            (Solver.un[j][i+1] - 2 * Solver.un[j][i] + Solver.un[j][i-1]) +
                            Solver.dt / Solver.dy**2 * 
                            (Solver.un[j+1][i] - 2 * Solver.un[j][i] + Solver.un[j-1][i])) + 
                            F(j+Solver.first,i,Solver.dt*Solver.stepcount) * Solver.dt)
        for i in xrange(Solver.ny):
            Solver.u[i][0]=0
        for i in xrange(Solver.ny):
            Solver.u[i][-1]=0
        ans=[]
        ans.append(list(Solver.u[1]))
        ans.append(list(Solver.u[-2]))
        Solver.interactions+=1
        return ans
    
    @staticmethod
    @expose
    def computev():
        for i in xrange(1,Solver.nx-1):
            for j in xrange(1,Solver.ny-1):
                Solver.v[j][i] = (Solver.vn[j][i] -
                            Solver.un[j][i] * Solver.dt / Solver.dx * 
                            (Solver.vn[j][i] - Solver.vn[j][i-1]) -
                            Solver.vn[j][i] * Solver.dt / Solver.dy * 
                            (Solver.vn[j][i] - Solver.vn[j-1][i]) -
                            Solver.dt / (2 * Solver.rho * Solver.dy) * 
                            (Solver.p[j+1][i] - Solver.p[j-1][i]) +
                            Solver.nu * (Solver.dt / Solver.dx**2 *
                            (Solver.vn[j][i+1] - 2 * Solver.vn[j][i] + Solver.vn[j][i-1]) +
                            Solver.dt / Solver.dy**2 * 
                            (Solver.vn[j+1][i] - 2 * Solver.vn[j][i] + Solver.vn[j-1][i])))
            for i in xrange(Solver.ny):
                Solver.v[i][0]=0
            for i in xrange(Solver.ny):
                Solver.v[i][-1]=0
        ans=[]
        ans.append(list(Solver.v[1]))
        ans.append(list(Solver.v[-2]))
        Solver.interactions+=1
        return ans
    @staticmethod
    @expose
    def getuv(uw,vw):
        Solver.u[0]=list(uw)
        Solver.v[0]=list(vw)
        Solver.interactions+=2
    @staticmethod
    @expose
    def getleftu(uw):
        Solver.u[0]=list(uw)
        Solver.interactions+=1
    @staticmethod
    @expose
    def getrightu(uw):
        Solver.u[-1]=list(uw)
        Solver.interactions+=1
    @staticmethod
    @expose
    def getleftv(vw):
        Solver.v[0]=list(vw)
        Solver.interactions+=1
    @staticmethod
    @expose
    def getrightv(vw):
        Solver.v[-1]=list(vw)
        Solver.interactions+=1
    @staticmethod
    @expose
    def giveansu():
        return Solver.u
    @staticmethod
    @expose
    def giveansv():
        return Solver.v
    @staticmethod
    @expose
    def getInteractions():
        return Solver.interactions
    @staticmethod
    @expose
    def qqq(n,A,x,l,r,st):
        if(st==0):
            Solver.n=n
            Solver.A=A
        xnew=[]
        for k in xrange(l,min(r,Solver.n)):
            xnew.append(float(Solver.A[k][Solver.n]))
            for i in xrange(Solver.n):
                if(not(i==k)):
                    xnew[k-l]=xnew[k-l]-Solver.A[k][i]*x[i]
            xnew[k-l]/=Solver.A[k][k]
        return xnew
    @staticmethod
    @expose
    def voidreduce(mapped):
        return True
    @staticmethod
    @expose
    def myreduce(mapped):
        print("reduce")
        output = []
        for q in mapped:
            print("reduce loop")
            w=q.value
            for i in xrange(len(w)-1):
                output.append(list(w[i]))
        output.append((mapped[-1].value)[-1])
        print("reduce done")
        
        return output 
    def checkInteractions(self):
        flag=True
        while(flag):
            s=[]
            for i in xrange(len(self.workers)):
                s.append((self.workers[i].getInteractions()).value)
            flag=False
            for i in xrange(len(self.workers)):
                if(not(s[i]==Solver.interactions)):

                    flag=True
                    break
    def read_input(self):
        f = open(self.input_file_name, 'r')
        W=int(f.readline())
        H=int(f.readline())
        nx=int(f.readline())
        ny=int(f.readline())
        rho=float((f.readline()))
        nu=float((f.readline()))
        dt=float((f.readline()))
        nit=int(f.readline())
        u=[]
        v=[]
        for i in xrange(ny):
            u.append([])
            v.append([])
            for j in xrange(nx):
                u[i].append(float(f.readline()))
                v[i].append(float(f.readline()))
        return (W,H,nx,ny,rho,nu,dt,nit,u,v)
    def write_output(self,W,H,nx,ny,u,v):
        f = open(self.output_file_name, 'w')
        f.write(str(W)+"\n")
        f.write(str(H)+"\n")
        f.write(str(nx)+"\n")
        f.write(str(ny)+"\n")
        for i in xrange(ny):
            for j in xrange(nx):
                f.write(str(u[i][j])+"\n")
                f.write(str(v[i][j])+"\n")
        f.close()
        print("output done") 