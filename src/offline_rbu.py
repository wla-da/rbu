import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, hilbert

YEAR_P=[((25,0),80),((26,0),40),((27,0),20),((28,0),10),((29,0),8),((30,0),4),((31,0),2),((32,0),1)]
HOUR_P=[((47,0),20),((48,0),10),((49,0),8),((50,0),4),((51,0),2),((52,0),1)]
MON_P =[((33,0),10),((34,0),8),((35,0),4),((36,0),2),((37,0),1)]
DOW_P =[((38,0),4),((39,0),2),((40,0),1)]
DAY_P =[((41,0),20),((42,0),10),((43,0),8),((44,0),4),((45,0),2),((46,0),1)]

def decode_fields(pos, MG, a, correct):
    secnum=(np.arange(pos.shape[0])-a)%60
    def maj(s,col=0):
        v=pos[secnum==s,col]; return int(v.mean()>0.5) if len(v) else 0
    def majm(s,col=0):
        v=MG[secnum==s,col]; return v.mean() if len(v) else 0.0
    corr={}
    def fix(pairs,par):
        tot=[(s,c) for (s,c),w in pairs]+[par]
        if sum(maj(s,c) for s,c in tot)%2:
            f=min(tot,key=lambda q:abs(majm(q[0],q[1])))
            corr[f]=1-maj(f[0],f[1])
    def getbit(s,col=0):
        return corr.get((s,col),maj(s,col))
    if correct:
        fix(YEAR_P,(54,1)); fix(HOUR_P,(57,1))
    pick=getbit if correct else maj
    def val(pairs): return sum(pick(s,c)*w for (s,c),w in pairs)
    return dict(year=val(YEAR_P),month=val(MON_P),dow=val(DOW_P),day=val(DAY_P),hour=val(HOUR_P))

def selftest():
    # синтез идеального кадра
    bit1={}
    def setf(pairs,value):
        for (s,c),w in pairs:
            b=1 if value>=w else 0
            if b: value-=w
            bit1[s]=b
    setf(YEAR_P,26); setf(MON_P,9); setf(DOW_P,5); setf(DAY_P,4); setf(HOUR_P,13)
    nsec=120; pos=np.zeros((nsec,10),int); 
    for k in range(nsec):
        s=k%60
        pos[k,2:8]=0; pos[k,9]=1; pos[k,8]=1 if s==59 else 0
        pos[k,0]=pos[k,1]=1 if s==0 else bit1.get(s,0)
    MG=pos*10-5
    a=60
    r=decode_fields(pos,MG,a,correct=False)
    ok=(r['year']==26 and r['month']==9 and r['day']==4 and r['hour']==13 and r['dow']==5)
    print('SELFTEST:', r, '->', 'OK' if ok else 'FAIL')
    return ok

def g(seg,f0):
    n=np.arange(len(seg)); return abs(np.sum(seg*np.exp(-1j*2*np.pi*f0*n/2000.)))

if len(sys.argv)>1 and sys.argv[1]=='selftest':
    raise SystemExit(0 if selftest() else 1)

WAV='../wav/iq_rbu5min.wav'
fs,d=wavfile.read(WAV)
mono=d.ndim<2
iq=hilbert(d.astype(np.float64)/32768.0) if mono else (d[:,0].astype(np.float64)+1j*d[:,1].astype(np.float64))/32768.0
N=len(iq)
print('='*64); print('файл:',WAV,'| режим:','МОНО' if mono else 'IQ',f'| t={N/fs:.1f}с')
Y=np.abs(np.fft.fft(iq*np.blackman(N))); fr=np.fft.fftfreq(N,1/fs)
band=(fr>4000)&(fr<6000) if not mono else (fr>20)&(fr<fs/2-500)
fb,Yb=fr[band],Y[band]; cands=[]
for i in np.argsort(-Yb):
    if all(abs(fb[i]-c)>50 for c in cands): cands.append(fb[i])
    if len(cands)>=6: break
def eval_fc(fc):
    bb=resample_poly(iq*np.exp(-1j*2*np.pi*fc*np.arange(N)/fs),2000,fs)
    M=len(bb);P=200;nper=M//P;best=None
    for off in range(0,200,20):
        bits=[];marg=[]
        for k in range(nper):
            seg=bb[k*P+off+4:k*P+off+196]
            if len(seg)<100:break
            p100=g(seg,100.)**2+g(seg,-100.)**2;p312=g(seg,312.5)**2+g(seg,-312.5)**2
            bits.append(1 if p312>p100 else 0);marg.append(np.log(p312/p100+1e-12))
        bits=np.array(bits);marg=np.array(marg)
        if len(bits)<30:continue
        for sh in range(10):
            p=np.roll(bits[:len(bits)//10*10].reshape(-1,10),-sh,axis=1)
            z=p[:,2:8].mean(axis=0);n9=p[:,9].mean();sc=n9-z.max()
            if best is None or sc>best[0]:best=(sc,off,sh,marg)
    return (bb,)+best if best else None
res=None;fc=None
for c in cands:
    r=eval_fc(c)
    if r and (res is None or r[1]>res[1]):res=r;fc=c
if res is None: raise SystemExit('несущая не найдена')
bb,sc,off,sh,marg=res
print(f'несущая {fc:.1f} счёт={sc:.2f} off={off} сдвиг={sh}')
nsec=len(marg)//10*10
pos=np.roll((marg>0).astype(int)[:nsec].reshape(-1,10),-sh,axis=1)
MG =np.roll(marg[:nsec].reshape(-1,10),-sh,axis=1)
a=None
for k in range(1,pos.shape[0]):
    if pos[k,0]==1 and pos[k,1]==1 and pos[k-1,8]==1 and pos[k-1,9]==1:
        a=k;break
if a is None:
    for k in range(1,pos.shape[0]):
        if pos[k,0]==1 and pos[k,1]==1:
            a=k;break
if a is None: raise SystemExit('якорь не найден')
print('якорь:',a)
r0=decode_fields(pos,MG,a,correct=False)
r1=decode_fields(pos,MG,a,correct=True)
f=lambda r:f"{r['day']:02d}.{r['month']:02d}.20{r['year']:02d} {r['hour']:02d}:xx ДН={r['dow']}"
print('СЫРОЙ  :',f(r0))
print('КОРРЕКТ:',f(r1))
for blk in range((pos.shape[0]-a)//60):
    F=pos[a+blk*60:a+blk*60+60]
    if len(F)<60:break
    b1=F[:,0]
    mm=sum(int(b1[s])*w for s,w in [(53,40),(54,20),(55,10),(56,8),(57,4),(58,2),(59,1)])
    print(f'рамка {blk}: минута {mm:02d}')
