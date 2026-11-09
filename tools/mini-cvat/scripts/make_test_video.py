import cv2, numpy as np
from skimage import data
rng=np.random.default_rng(1)
src=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2BGR)
FACE=np.array([178,63,90,113],float)          # box mặt trên ảnh gốc 512x512
W,H,N=960,540,90
bg=cv2.resize((rng.random((54,96,3))*255).astype(np.uint8),(W,H),interpolation=cv2.INTER_CUBIC)
vw=cv2.VideoWriter('test.mp4',cv2.VideoWriter_fourcc(*'mp4v'),25,(W,H))
gt=open('gt.txt','w')
def paste(fr,img,s,x,y,flip=False):
    p=cv2.resize(src,None,fx=s,fy=s); 
    if flip: p=p[:,::-1]
    h,w=p.shape[:2]; x0,y0=int(x),int(y)
    xa,ya,xb,yb=max(0,x0),max(0,y0),min(W,x0+w),min(H,y0+h)
    img[ya:yb,xa:xb]=p[ya-y0:yb-y0,xa-x0:xb-x0]
    fb=FACE*s; fx=fb[0] if not flip else 512*s-fb[0]-fb[2]
    return np.array([x0+fx,y0+fb[1],fb[2],fb[3]])
for t in range(1,N+1):
    img=bg.copy()
    b2=paste(t,img,0.22,880-7*t,300,flip=True)              # người 2: mặt nhỏ ~20x25, đi sang trái
    b1=paste(t,img,0.75-0.004*t,20+5*t,40)                  # người 1: mặt to dần nhỏ, đi sang phải
    if 20<=t<=27: img=cv2.filter2D(img,-1,np.ones((1,21))/21)            # motion blur
    if 36<=t<=42:                                                        # che nửa mặt người 1
        x,y,w,h=b1.astype(int); img[y:y+h,x:x+w//2+8]=(120,120,120)
    if 50<=t<=56: img=np.clip(img*0.25+rng.normal(0,18,img.shape),0,255).astype(np.uint8)  # tối + nhiễu
    img=np.clip(img+rng.normal(0,6,img.shape),0,255).astype(np.uint8)
    vw.write(img)
    for i,b in ((1,b1),(2,b2)):
        if b[0]+b[2]>0 and b[0]<W: gt.write(f"{t},{i},{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f},1,-1,-1\n")
vw.release(); gt.close(); print('ok')
