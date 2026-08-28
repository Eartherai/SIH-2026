#!/usr/bin/env python3
"""Real failure cases, from actual inference. No curation of successes."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aquashield.detection.detector import Detector
from aquashield.detection.boxes import xywhn_to_xyxy
from aquashield.evaluation.matching import match

S = Path("data/processed/milco_nombo_yolo/test")
det = Detector("models/aquashield_primary.pt", conf=0.05)

def load(name):
    g = cv2.imread(str(S/"images"/name), cv2.IMREAD_GRAYSCALE)
    lp = (S/"labels"/name).with_suffix(".txt")
    rows = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
    gt = xywhn_to_xyxy(np.array([[float(v) for v in r] for r in rows],np.float32)[:,1:5],
                       g.shape[1],g.shape[0]) if rows else np.zeros((0,4),np.float32)
    return g, gt

def speckle(img, sigma, seed):
    rng=np.random.default_rng(seed)
    gm=rng.gamma(1.0/sigma, sigma, img.shape)
    return np.clip(img.astype(np.float32)*gm,0,255).astype(np.uint8)

def tile(g, title, sub, dets, gt, color=(0,200,255)):
    b=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for x0,y0,x1,y1 in gt.astype(int):
        cv2.rectangle(b,(x0-3,y0-3),(x1+3,y1+3),(255,255,255),2)
    for d in dets:
        x0,y0,x1,y1=[int(v) for v in d]
        cv2.rectangle(b,(x0,y0),(x1,y1),color,2)
    b=cv2.resize(b,(300,300),interpolation=cv2.INTER_AREA)
    bar=np.full((40,300,3),25,np.uint8)
    cv2.putText(bar,title,(6,17),cv2.FONT_HERSHEY_SIMPLEX,0.42,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(bar,sub,(6,33),cv2.FONT_HERSHEY_SIMPLEX,0.38,(140,180,255),1,cv2.LINE_AA)
    return np.vstack([bar,b])

def dets_of(g):
    r=det.detect(g,640,128)
    return ([d.box_xyxy for d in r.detections],
            np.array([d.box_xyxy for d in r.detections],np.float32) if r.detections else np.zeros((0,4),np.float32),
            np.array([d.raw_score for d in r.detections],np.float32) if r.detections else np.zeros(0,np.float32))

# 1. LARGE TARGET MISSED — pick the frame with the biggest GT box
biggest=None
for ip in sorted((S/"images").glob("*.jpg")):
    g,gt=load(ip.name)
    if len(gt)==0: continue
    a=((gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1])).max()
    if biggest is None or a>biggest[0]: biggest=(a,ip.name)
g,gt=load(biggest[1]); bl,db,ds=dets_of(g)
m=match(db,ds,gt,0.3)
t1=tile(g,"FAIL: large target missed",f"{biggest[1]} GT area {biggest[0]:.0f}px2, TP={m.tp}",bl,gt)

# 2. NATURAL-SEABED FALSE POSITIVE — empty frame with most detector alarms
worst=None
for ip in sorted((S/"images").glob("*.jpg")):
    g,gt=load(ip.name)
    if len(gt)>0: continue
    bl,_,_=dets_of(g)
    if worst is None or len(bl)>worst[0]: worst=(len(bl),ip.name)
    if worst[0]>=4: break
g,gt=load(worst[1]); bl,_,_=dets_of(g)
t2=tile(g,"FAIL: natural-seabed false positives",f"{worst[1]} empty frame, {len(bl)} alarms",bl,gt,(70,90,240))

# 3. SPECKLE COLLAPSE — a frame the model gets clean but loses under speckle
demo=None
for ip in sorted((S/"images").glob("*.jpg")):
    g,gt=load(ip.name)
    if len(gt)==0: continue
    _,db,ds=dets_of(g)
    if match(db,ds,gt,0.3).tp>0: demo=ip.name; break
g,gt=load(demo); bl,_,_=dets_of(g)
t3=tile(g,"clean: target detected",f"{demo}",bl,gt)
gs=speckle(g,0.3,0); bls,_,_=dets_of(gs)
t4=tile(gs,"FAIL: same frame + speckle",f"sigma=0.3, {len(bls)} detections",bls,gt)

top=np.hstack([t1,np.full((t1.shape[0],6,3),30,np.uint8),t2])
bot=np.hstack([t3,np.full((t3.shape[0],6,3),30,np.uint8),t4])
sep=np.full((6,top.shape[1],3),30,np.uint8)
out=np.vstack([top,sep,bot])
cap=np.full((54,out.shape[1],3),18,np.uint8)
cv2.putText(cap,"AQUA-SHIELD failure gallery - real inference, held-out surveys (raw E04 model)",
            (10,22),cv2.FONT_HERSHEY_SIMPLEX,0.46,(235,235,235),1,cv2.LINE_AA)
cv2.putText(cap,"white=ground truth  orange=detection  blue=false positive.  Failures shown deliberately.",
            (10,44),cv2.FONT_HERSHEY_SIMPLEX,0.4,(150,150,150),1,cv2.LINE_AA)
final=np.vstack([out,cap])
Path("docs/images").mkdir(parents=True,exist_ok=True)
cv2.imwrite("docs/images/failure_gallery.png",final)
cv2.imwrite("outputs/failure_gallery/gallery.png",final)
print("wrote docs/images/failure_gallery.png")
print(f"  large-target-missed: {biggest[1]} (area {biggest[0]:.0f}px2)")
print(f"  false-positive frame: {worst[1]} ({worst[0]} alarms on empty seabed)")
print(f"  speckle: {demo} clean {len(bl)} dets -> speckled {len(bls)} dets")
