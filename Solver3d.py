from Pyro4 import expose
from random import randint
import random
import time
import copy
class Node:
    def __init__(self,u,v,w,first):
        self.u=u
        self.v=v
        self.w=w
        self.first=first
def F(j,i,k,t):
    if(i>5 and i<15 and j>5 and j<15 and k>5 and k<15 and t<0.15):
        return 10
    else:
        return 0
class Solver:
    v=[]
    u=[]
    w=[]
    p=[]
    vn=[]
    un=[]
    wn=[]
    pn=[]
    b=[]
    ny=0
    nx=0
    nz=0
    rho=0
    nu=0
    num=0
    dt=0
    dx=0
    dy=0
    dz=0
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
        (L,W,H,nx,ny,nz,rho,nu,dt,nit,u,v,w) = self.read_input()
        dx = (L+0.1-0.1) / (nx - 1)
        dy = (W+0.1-0.1) / (ny - 1)
        dz = (H+0.1-0.1) / (nz - 1)
        un = []
        kw=len(self.workers)
        step=ny//kw+1
        uw=[]
        vw=[]
        ww=[]
        nodes=[]
        first=0
        for i in xrange(ny):
            uw.append(list(u[i]))
            vw.append(list(v[i]))
            ww.append(list(w[i]))
            if((i+1)%step==0):
                if(len(uw)<=0):
                    raise Exception(str(i)+"Wrong size of part init")
                nodes.append(Node(uw,vw,ww,first))
                uw=[]
                uw.append(list(u[i-1]))
                uw.append(list(u[i]))
                vw=[]
                vw.append(list(v[i-1]))
                vw.append(list(v[i]))
                ww=[]
                ww.append(list(w[i-1]))
                ww.append(list(w[i]))
                first=i-1
        nodes.append(Node(uw,vw,ww,first))
        c=[]
        for i in xrange(kw):
            if(len(nodes[i].u)<=1):
                raise Exception(str(i)+"Wrong size of part initgetnode")
            c.append(self.workers[i].getnode(nodes[i].u,nodes[i].v,nodes[i].w,nodes[i].first,rho,nu,dt,dx,dy,dz,i,step))
            
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
                self.checkInteractions()
                for i in xrange(kw):
                    c.append(self.workers[i].getp([pborders[i][1],pborders[i+1][0]]))
                Solver.interactions+=1
                self.checkInteractions()
                c
            
            uborders=[]
            vborders=[]
            wborders=[]
            ubordersreturns=[]
            vbordersreturns=[]
            wbordersreturns=[]
            for i in xrange(kw):
                ubordersreturns.append((self.workers[i].computeu()))
            for i in xrange(kw):
                vbordersreturns.append((self.workers[i].computev()))
            for i in xrange(kw):
                wbordersreturns.append((self.workers[i].computew()))
            for i in xrange(kw):
                uborders.append(ubordersreturns[i].value)
                vborders.append(vbordersreturns[i].value)
                wborders.append(wbordersreturns[i].value)
            Solver.interactions+=3
            self.checkInteractions()

            zerovector=[]
            for j in xrange(nx):
                zerovector.append([0 for i in xrange(nz)])
            c=[]
            c.append(self.workers[0].getleftu(zerovector))
            
            c.append(self.workers[0].getleftv(zerovector))
            
            c.append(self.workers[0].getleftw(zerovector))
            
            for i in xrange(1,kw):
                c.append(self.workers[i-1].getrightu(uborders[i][0]))
            for i in xrange(1,kw):
                c.append(self.workers[i-1].getrightv(vborders[i][0]))
            for i in xrange(1,kw):
                c.append(self.workers[i-1].getrightw(vborders[i][0]))
            for i in xrange(1,kw):
                c.append(self.workers[i].getleftu(uborders[i-1][1]))
            for i in xrange(1,kw):
                c.append(self.workers[i].getleftv(vborders[i-1][1]))
            for i in xrange(1,kw):
                c.append(self.workers[i].getleftw(wborders[i-1][1]))
            
            c.append(self.workers[-1].getrightu(zerovector))
            c.append(self.workers[-1].getrightv(zerovector))
            c.append(self.workers[-1].getrightw(zerovector))
            Solver.interactions+=6
            self.checkInteractions()
            c
            stepcount+=1 
        umapped=[]
        vmapped=[]
        wmapped=[]
        for i in xrange(kw):
            umapped.append(self.workers[i].giveansu())
        for i in xrange(kw):
            vmapped.append(self.workers[i].giveansv())
        for i in xrange(kw):
            wmapped.append(self.workers[i].giveansw())
        
        
        uans=self.myreduce(umapped)
        
        vans=self.myreduce(vmapped)

        wans=self.myreduce(wmapped)
        
        self.write_output(L,W,H,nx,ny,nz,uans,vans,wans)
    @staticmethod
    @expose
    def getnode(u,v,w,first,rho,nu,dt,dx,dy,dz,num,step):
        Solver.u=u
        Solver.v=v
        Solver.w=w
        Solver.first=first
        Solver.rho=rho
        Solver.nu=nu
        Solver.dt=dt
        Solver.dx=dx
        Solver.dy=dy
        Solver.dz=dz
        Solver.num=num
        Solver.step=step
        Solver.ny=len(u)
        Solver.nx=len(u[0])
        Solver.nz=len(u[0][0])
        if(Solver.ny<=1):
            raise Exception(str(i)+"Wrong size of part getnode")
        for i in xrange(Solver.ny):
            Solver.un.append([[0 for k in xrange(Solver.nz)] for k in xrange(Solver.nx)])    
        Solver.vn = []
        for i in xrange(Solver.ny):
            Solver.vn.append([[0 for j in xrange(Solver.nz)] for k in xrange(Solver.nx)])
        Solver.wn = []
        for i in xrange(Solver.ny):
            Solver.wn.append([[0 for j in xrange(Solver.nz)] for k in xrange(Solver.nx)])
        Solver.p = []
        for i in xrange(Solver.ny):
            Solver.p.append([[0 for j in xrange(Solver.nz)] for k in xrange(Solver.nx)])
        Solver.pn = []
        for i in xrange(Solver.ny):
            Solver.pn.append([[0 for j in xrange(Solver.nz)] for k in xrange(Solver.nx)])
        Solver.b = []
        for i in xrange(Solver.ny):
            Solver.b.append([[0 for j in xrange(Solver.nz)] for k in xrange(Solver.nx)])
        Solver.interactions+=1
    @staticmethod
    @expose
    def computeb():
        if(Solver.ny<=1):
            raise Exception(str(ny)+"Wrong size of part computeb")
        Solver.un = copy.deepcopy(Solver.u)
        Solver.vn = copy.deepcopy(Solver.v)
        Solver.wn = copy.deepcopy(Solver.w)
        Solver.b = []
        for i in xrange(Solver.ny):
            Solver.b.append([[0 for j in xrange(Solver.nz)] for k in xrange(Solver.nx)])
    
        for i in xrange(1,Solver.nx-1):
            for j in xrange(1,Solver.ny-1):
                for k in xrange(1,Solver.nz-1):
                    Solver.b[j][i][k] = (Solver.rho * (1 / Solver.dt * ((Solver.u[j][i+1][k] - Solver.u[j][i-1][k]) / (2 * Solver.dx) +
                                            (Solver.v[j+1][i][k] - Solver.v[j-1][i][k]) / (2 * Solver.dy)+
                                            (Solver.w[j][i][k+1]-Solver.w[j][i][k-1])/(2*Solver.dz)) -
                                    ((Solver.u[j][i+1][k] - Solver.u[j][i-1][k]) / (2 * Solver.dx))**2 -
                                    2 * ((Solver.u[j+1][i][k] - Solver.u[j-1][i][k]) / (2 * Solver.dy) *
                                        (Solver.v[j][i+1][k] - Solver.v[j][i-1][k]) / (2 * Solver.dx))-
                                    ((Solver.v[j+1][i][k] - Solver.v[j-1][i][k]) / (2 * Solver.dy))**2-
                                    ((Solver.w[j][i][k+1]-Solver.w[j][i][k-1])/(2*Solver.dz))**2-
                                    2 * ((Solver.w[j+1][i][k] - Solver.w[j-1][i][k]) / (2 * Solver.dy) *
                                        (Solver.v[j][i][k+1] - Solver.v[j][i][k-1]) / (2 * Solver.dz))-
                                        2 * ((Solver.u[j][i][k+1] - Solver.u[j][i][k-1]) / (2 * Solver.dz) *
                                        (Solver.w[j][i+1][k] - Solver.w[j][i-1][k]) / (2 * Solver.dx))))
        Solver.interactions+=1
    @staticmethod
    @expose
    def computep():
        if(Solver.ny<=1):
            raise Exception(str(len(Solver.p))+"Wrong size of part computep")
        Solver.pn = copy.deepcopy(Solver.p)
        
        for i in xrange(1,Solver.nx-1):
            for j in xrange(1,Solver.ny-1):
                for k in xrange(1,Solver.nz-1):
                    Solver.p[j][i][k] = (((Solver.pn[j][i+1][k] + Solver.pn[j][i-1][k]) * (Solver.dy**2)*(Solver.dz**2) +
                                (Solver.pn[j+1][i][k] + Solver.pn[j-1][i][k]) * (Solver.dx**2)*(Solver.dz**2)+
                                (Solver.pn[j][i][k+1]+Solver.pn[j][i][k-1])*(Solver.dx**2)*(Solver.dy**2)) /
                                (2 * (Solver.dx**2 + Solver.dy**2+Solver.dz**2)) -
                                Solver.dx**2 * Solver.dy**2 * Solver.dz**2/ (2 * ((Solver.dx**2)*(Solver.dy**2) + (Solver.dy**2)*(Solver.dz**2) + (Solver.dx**2)*(Solver.dz**2))) * Solver.b[j][i][k])
        for i in xrange(Solver.ny):
            for j in xrange(Solver.nx):
                Solver.p[i][j][-1]=Solver.p[i][j][-2]
                Solver.p[i][j][0]=Solver.p[i][j][1]
        for i in xrange(Solver.ny):
            for k in xrange(Solver.nz):
                Solver.p[i][-1][k]=Solver.p[i][-2][k]
                Solver.p[i][0][k]=Solver.p[i][1][k]
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
                for k in xrange(1,Solver.nz-1):
                    Solver.u[j][i][k] = (Solver.un[j][i][k] -
                            Solver.un[j][i][k] * Solver.dt / Solver.dx * 
                            (Solver.un[j][i][k] - Solver.un[j][i-1][k]) -
                            Solver.vn[j][i][k] * Solver.dt / Solver.dy * 
                            (Solver.un[j][i][k] - Solver.un[j-1][i][k]) -
                            Solver.wn[j][i][k] * Solver.dt / Solver.dz *
                            (Solver.un[j][i][k] - Solver.un[j][i][k-1])-
                            Solver.dt / (2 * Solver.rho * Solver.dx) * 
                            (Solver.p[j][i+1][k] - Solver.p[j][i-1][k]) +
                            Solver.nu * (Solver.dt / Solver.dx**2 * 
                            (Solver.un[j][i+1][k] - 2 * Solver.un[j][i][k] + Solver.un[j][i-1][k]) +
                            Solver.dt / Solver.dy**2 * 
                            (Solver.un[j+1][i][k] - 2 * Solver.un[j][i][k] + Solver.un[j-1][i][k]) + 
                            Solver.dt / Solver.dz**2 * 
                            (Solver.un[j][i][k+1] - 2 * Solver.un[j][i][k] + Solver.un[j][i][k-1])) + 
                            F(j+Solver.first,i,k,Solver.dt*Solver.stepcount) * Solver.dt)
        for i in xrange(Solver.ny):
            for k in xrange(Solver.nz):
                Solver.u[i][0][k]=0
                Solver.u[i][-1][k]=0
        for i in xrange(Solver.ny):
            for j in xrange(Solver.nx):
                Solver.u[i][j][0]=0
                Solver.u[i][j][-1]=0
        
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
                for k in xrange(1,Solver.nz-1):
                    Solver.v[j][i][k] = (Solver.vn[j][i][k] -
                            Solver.un[j][i][k] * Solver.dt / Solver.dx * 
                            (Solver.vn[j][i][k] - Solver.vn[j][i-1][k]) -
                            Solver.vn[j][i][k] * Solver.dt / Solver.dy * 
                            (Solver.vn[j][i][k] - Solver.vn[j-1][i][k]) -
                            Solver.wn[j][i][k] * Solver.dt / Solver.dz *
                            (Solver.vn[j][i][k] - Solver.vn[j][i][k-1])-
                            Solver.dt / (2 * Solver.rho * Solver.dy) * 
                            (Solver.p[j+1][i][k] - Solver.p[j-1][i][k]) +
                            Solver.nu * (Solver.dt / Solver.dx**2 * 
                            (Solver.vn[j][i+1][k] - 2 * Solver.vn[j][i][k] + Solver.vn[j][i-1][k]) +
                            Solver.dt / Solver.dy**2 * 
                            (Solver.vn[j+1][i][k] - 2 * Solver.vn[j][i][k] + Solver.vn[j-1][i][k]) + 
                            Solver.dt / Solver.dz**2 * 
                            (Solver.vn[j][i][k+1] - 2 * Solver.vn[j][i][k] + Solver.vn[j][i][k-1]))+ 
                            F(j+Solver.first,i,k,Solver.dt*Solver.stepcount) * Solver.dt)
        for i in xrange(Solver.ny):
            for k in xrange(Solver.nz):
                Solver.v[i][0][k]=0
                Solver.v[i][-1][k]=0
        for i in xrange(Solver.ny):
            for j in xrange(Solver.nx):
                Solver.v[i][j][0]=0
                Solver.v[i][j][-1]=0
        ans=[]
        ans.append(list(Solver.v[1]))
        ans.append(list(Solver.v[-2]))
        Solver.interactions+=1
        return ans
    @staticmethod
    @expose
    def computew():
        for i in xrange(1,Solver.nx-1):
            for j in xrange(1,Solver.ny-1):
                for k in xrange(1,Solver.nz-1):
                    Solver.w[j][i][k] = (Solver.wn[j][i][k] -
                            Solver.un[j][i][k] * Solver.dt / Solver.dx * 
                            (Solver.wn[j][i][k] - Solver.wn[j][i-1][k]) -
                            Solver.vn[j][i][k] * Solver.dt / Solver.dy * 
                            (Solver.wn[j][i][k] - Solver.wn[j-1][i][k]) -
                            Solver.wn[j][i][k] * Solver.dt / Solver.dz *
                            (Solver.wn[j][i][k] - Solver.wn[j][i][k-1])-
                            Solver.dt / (2 * Solver.rho * Solver.dy) * 
                            (Solver.p[j][i][k+1] - Solver.p[j][i][k-1]) +
                            Solver.nu * (Solver.dt / Solver.dx**2 * 
                            (Solver.wn[j][i+1][k] - 2 * Solver.wn[j][i][k] + Solver.wn[j][i-1][k]) +
                            Solver.dt / Solver.dy**2 * 
                            (Solver.wn[j+1][i][k] - 2 * Solver.wn[j][i][k] + Solver.wn[j-1][i][k]) + 
                            Solver.dt / Solver.dz**2 * 
                            (Solver.wn[j][i][k+1] - 2 * Solver.wn[j][i][k] + Solver.wn[j][i][k-1]))+ 
                            F(j+Solver.first,i,k,Solver.dt*Solver.stepcount) * Solver.dt)
        for i in xrange(Solver.ny):
            for k in xrange(Solver.nz):
                Solver.w[i][0][k]=0
                Solver.w[i][-1][k]=0
        for i in xrange(Solver.ny):
            for j in xrange(Solver.nx):
                Solver.w[i][j][0]=0
                Solver.w[i][j][-1]=0
        ans=[]
        ans.append(list(Solver.w[1]))
        ans.append(list(Solver.w[-2]))
        Solver.interactions+=1
        return ans
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
    def getleftw(ww):
        Solver.w[0]=list(ww)
        Solver.interactions+=1
    @staticmethod
    @expose
    def getrightw(ww):
        Solver.w[-1]=list(ww)
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
    def giveansw():
        return Solver.w
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
        L=int(f.readline())
        W=int(f.readline())
        H=int(f.readline())
        nx=int(f.readline())
        ny=int(f.readline())
        nz=int(f.readline())
        rho=float((f.readline()))
        nu=float((f.readline()))
        dt=float((f.readline()))
        nit=int(f.readline())
        u=[]
        v=[]
        w=[]
        for i in xrange(ny):
            u.append([])
            v.append([])
            w.append([])
            for j in xrange(nx):
                u[i].append([])
                v[i].append([])
                w[i].append([])
                for k in xrange(nz):
                    u[i][j].append(float(f.readline()))
                    v[i][j].append(float(f.readline()))
                    w[i][j].append(float(f.readline()))
        return (L,W,H,nx,ny,nz,rho,nu,dt,nit,u,v,w)
    def write_output(self,L,W,H,nx,ny,nz,u,v,w):
        f = open(self.output_file_name, 'w')
        f.write(str(L)+"\n")
        f.write(str(W)+"\n")
        f.write(str(H)+"\n")
        f.write(str(nx)+"\n")
        f.write(str(ny)+"\n")
        f.write(str(nz)+"\n")
        for i in xrange(ny):
            for j in xrange(nx):
                for k in xrange(nz):
                    f.write(str(u[i][j][k])+"\n")
                    f.write(str(v[i][j][k])+"\n")
                    f.write(str(w[i][j][k])+"\n")

        f.close()
        print("output done") 