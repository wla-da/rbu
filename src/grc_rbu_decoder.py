import numpy as np
from gnuradio import gr
from scipy.signal import resample_poly

Y ={25:80,26:40,27:20,28:10,29:8,30:4,31:2,32:1}
MO={33:10,34:8,35:4,36:2,37:1}
W ={38:4,39:2,40:1}
D ={41:20,42:10,43:8,44:4,45:2,46:1}
H ={47:20,48:10,49:8,50:4,51:2,52:1}
MI={53:40,54:20,55:10,56:8,57:4,58:2,59:1}
TJD={18:8000,19:4000,20:2000,21:1000,22:800,23:400,24:200,25:100,
     26:80,27:40,28:20,29:10,30:8,31:4,32:2,33:1}
PAR={49:'P1',50:'P2',53:'P3',54:'P4',55:'P5',56:'P6',57:'P7',58:'P8'}
RES0={1,2,8,9,10,16,17,24}
RES2=set(range(34,49))|{17,51,52,59}
DOW=['Пн','Вт','Ср','Чт','Пт','Сб','Вс']

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self,name='RBU_decoder',
            in_sig=[np.complex64],out_sig=[np.complex64])
        self.raw=np.zeros(0,np.complex64); self.bbbuf=np.zeros(0,np.complex64)
        self.fc=None; self.chunk=0
        self.elems=[]; self.rawrows=[]; self.rows=[]
        self.sh=None; self.anchor=None
        self.h1={}; self.h2={}; self.latest1={}; self.min_latch=None
        n=np.arange(192)
        self.e100p=np.exp(-1j*2*np.pi*100.0*n/2000.); self.e100m=self.e100p.conj()
        self.e312p=np.exp(-1j*2*np.pi*312.5*n/2000.); self.e312m=self.e312p.conj()
        print('[RBU] t=  0.0s запущен')
    def work(self,inp,out):
        x=inp[0]; out[0][:]=x
        self.raw=np.concatenate([self.raw,x])
        if self.fc is None and len(self.raw)>=192000:
            sp=np.abs(np.fft.fft(self.raw[:192000]*np.blackman(192000)))
            fr=np.fft.fftfreq(192000,1/192000.); m=(fr>4000)&(fr<6000)
            self.fc=fr[m][np.argmax(sp[m])]; print('[RBU] несущая %.1f'%self.fc)
        while len(self.raw)>=192000:
            ch=self.raw[:192000]; self.raw=self.raw[192000:]
            t=(self.chunk+np.arange(192000))/192000.0
            self.bbbuf=np.concatenate([self.bbbuf,resample_poly(ch*np.exp(-1j*2*np.pi*self.fc*t),2000,192000)])
            self.chunk+=1
        while self.fc is not None and len(self.bbbuf)>=200:
            seg=self.bbbuf[4:196]; self.bbbuf=self.bbbuf[200:]
            if len(seg)<192: continue
            p100=abs(np.sum(seg*self.e100p))**2+abs(np.sum(seg*self.e100m))**2
            p312=abs(np.sum(seg*self.e312p))**2+abs(np.sum(seg*self.e312m))**2
            self.elems.append(1 if p312>p100 else 0)
            if len(self.elems)>=10:
                self.on_row(self.elems[:10]); self.elems=self.elems[10:]
        return len(x)
    def on_row(self,raw):
        if self.sh is None:
            self.rawrows.append(raw)
            if len(self.rawrows)>=30:
                best=None
                for sh in range(10):
                    p=np.roll(np.array(self.rawrows),-sh,axis=1)
                    sc=p[:,9].mean()-p[:,2:7].mean(axis=0).max()
                    if best is None or sc>best[0]: best=(sc,sh)
                self.sh=best[1]
                self.rows=[list(np.roll(r,-self.sh)) for r in self.rawrows]
                print('[RBU] t=%5.1fs sh=%d'%(self.chunk,self.sh))
                for i,r in enumerate(self.rows):
                    if r[0]==1 and r[1]==1: self.anchor=i;break
                if self.anchor is not None: print('[RBU] ЯКОРЬ %d'%self.anchor); self.rebuild()
            return
        row=list(np.roll(raw,-self.sh)); self.rows.append(row)
        if self.anchor is None:
            if row[0]==1 and row[1]==1:
                self.anchor=len(self.rows)-1; print('[RBU] ЯКОРЬ %d'%self.anchor); self.rebuild()
            return
        self.ingest(len(self.rows)-1); self.status()
    def rebuild(self):
        self.h1={}; self.h2={}; self.latest1={}; self.min_latch=None
        for i in range(len(self.rows)): self.ingest(i)
        self.status()
    def ingest(self,i):
        s=(i-self.anchor)%60; r=self.rows[i]
        for h,v in ((self.h1,r[0]),(self.h2,r[1])):
            h.setdefault(s,[]).append(v); h[s]=h[s][-8:]
        self.latest1[s]=r[0]
        if s==0 and all(q in self.latest1 for q in range(53,60)):
            self.min_latch=sum(self.latest1[q]*w for q,w in MI.items())
    def conf(self,s,col):
        h=self.h1 if col==0 else self.h2
        if s not in h or not h[s]: return None,0.0
        m=np.mean(h[s]); return int(m>0.5),abs(m-0.5)
    def group(self,members):
        bits={};cf={}
        for sc in members:
            b,c=self.conf(*sc)
            if b is None: return None
            bits[sc]=b;cf[sc]=c
        if sum(bits.values())%2:
            f=min(cf,key=cf.get); bits[f]^=1
        return bits
    def field(self,pairs,psec):
        g=self.group([(s,0) for s,_ in pairs]+[(psec,1)])
        if g is not None: return sum(g[(s,0)]*w for s,w in pairs)
        v=0
        for s,w in pairs:
            b,_=self.conf(s,0)
            if b is None: return None
            v|=b*w
        return v
    def valid_of(self,s,row):
        ok=all(row[i]==0 for i in range(2,7)) and row[9]==1
        if s is None: return ok
        if s==0: ok=ok and row[0]==1 and row[1]==1 and row[7]==1 and row[8]==1
        else:    ok=ok and row[7]==0 and row[8]==0
        if s in RES0: ok=ok and row[0]==0
        if s in RES2: ok=ok and row[1]==0
        return ok
    def tag1(self,s,b):
        for M,pre in ((Y,'Y'),(MO,'M'),(W,'W'),(D,'D'),(H,'H'),(MI,'m')):
            if s in M: return '%s%d=%d'%(pre,M[s],b)
        if s==0:return 'старт1'; 
        if 3<=s<=7:return 'DUT+1'; 
        if 11<=s<=15:return 'DUT-1'; 
        if 18<=s<=23:return 'TJD1'
        return 'резерв1'
    def tag2(self,s,b):
        if s in PAR: return '%s=%d'%(PAR[s],b)
        if s in TJD: return 'TJD%d=%d'%(TJD[s],b)
        if 1<=s<=8: return 'DUT1+=%d'%b
        if 9<=s<=16:return 'DUT1-=%d'%b
        if s==0:return 'старт2'
        return 'резерв2'
    def status(self):
        t=self.chunk
        s=(len(self.rows)-1-self.anchor)%60 if self.anchor is not None else None
        yr=self.field(list(Y.items()),54); mo=self.field(list(MO.items()),55)
        dd=self.field(list(D.items()),56); hh=self.field(list(H.items()),57)
        dw=self.field(list(W.items()),55)
        Ys=('%04d'%(2000+yr)) if yr is not None else 'YYYY'
        Mo=('%02d'%mo) if mo is not None else 'MM'
        Ds=('%02d'%dd) if dd is not None else 'DD'
        Hs=('%02d'%hh) if hh is not None else 'HH'
        Mi=('%02d'%self.min_latch) if self.min_latch is not None else 'MM'
        Ss=('%02d'%s) if s is not None else 'SS'
        row=self.rows[-1]
        print('[RBU] t=%6.1fs %s.%s.%s %s:%s:%s | кадр:%s %s | б1:%s б2:%s'%(
            t,Ys,Mo,Ds,Hs,Mi,Ss,''.join(map(str,row)),
            'V' if self.valid_of(s,row) else '-',
            self.tag1(s,row[0]) if s is not None else '',
            self.tag2(s,row[1]) if s is not None else ''))
