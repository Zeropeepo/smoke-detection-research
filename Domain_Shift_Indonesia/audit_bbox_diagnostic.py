"""Diagnose fixed-threshold bbox matches; indicators, not verified root causes."""
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent / "arsip" / "hasil_yolo11n_26n_20261123"
OUT = BASE / "analysis"
LABELS = BASE / "prepared/split/labels/test"
CLASSES = {0: "smoke", 1: "fire"}
CONF, IOU, NEAR = .25, .50, .10

def read_boxes(path, pred=False):
    if not path.exists(): return []
    rows=[]
    for line in path.read_text().splitlines():
        v=line.split()
        if v:
            assert len(v)==(6 if pred else 5), (path,line)
            rows.append({"cls":int(v[0]),"xywh":tuple(map(float,v[1:5])),"confidence":float(v[5]) if pred else None})
    return rows

def iou(a,b):
    x,y,w,h=a; X,Y,W,H=b
    inter=max(0,min(x+w/2,X+W/2)-max(x-w/2,X-W/2))*max(0,min(y+h/2,Y+H/2)-max(y-h/2,Y-H/2))
    union=w*h+W*H-inter
    return inter/union if union>0 else 0.

def diagnostic(gt,pred,cls):
    targets=[g for g in gt if g['cls']==cls]
    own=sorted((p for p in pred if p['cls']==cls and p['confidence']>=CONF),key=lambda p:-p['confidence'])
    matched=set(); fp=[]
    for p in own:
        options=[(iou(p['xywh'],g['xywh']),i) for i,g in enumerate(targets) if i not in matched]
        best=max(options,default=(0.,None))
        if best[1] is not None and best[0]>=IOU: matched.add(best[1]);continue
        own_i=max((iou(p['xywh'],g['xywh']) for g in targets),default=0.)
        other=max((iou(p['xywh'],g['xywh']) for g in gt if g['cls']!=cls),default=0.)
        kind='competition_or_duplicate' if own_i>=IOU else ('possible_class_confusion' if other>=IOU else ('localization_overlap_0.1_0.5' if own_i>=NEAR else 'no_nearby_gt'))
        fp.append({'type':kind,'best_iou_same':own_i,'best_iou_other':other,'confidence':p['confidence'],'bbox':p['xywh']})
    fn=[]
    for i,g in enumerate(targets):
        if i in matched:continue
        same=[(iou(g['xywh'],p['xywh']),p['confidence']) for p in pred if p['cls']==cls]
        high=max((a for a in same if a[1]>=CONF),default=(0.,0.))
        low=max((a for a in same if a[1]<CONF),default=(0.,0.))
        other=max((iou(g['xywh'],p['xywh']) for p in pred if p['cls']!=cls and p['confidence']>=CONF),default=0.)
        kind='competition_or_duplicate' if high[0]>=IOU else ('below_confidence_possible' if low[0]>=IOU else ('possible_class_confusion' if other>=IOU else ('localization_overlap_0.1_0.5' if high[0]>=NEAR else 'no_nearby_prediction')))
        fn.append({'type':kind,'best_iou_same':high[0],'best_iou_other':other,'best_iou_below_conf':low[0],'bbox':g['xywh']})
    return len(matched),fp,fn

def main():
    rows=[];counts=[]
    for model in ['yolo11n','yolo26n']:
      for stage in ['before','after']:
        predroot=BASE/'results_adapt'/model/stage/'plots/labels'
        if not predroot.is_dir():raise FileNotFoundError(predroot)
        for f in sorted(LABELS.glob('*.txt')):
          gt=read_boxes(f); pred=read_boxes(predroot/f.name,True)
          for cls,name in CLASSES.items():
            tp,fp,fn=diagnostic(gt,pred,cls)
            counts.append({'model':model,'stage':stage,'class':name,'image_stem':f.stem,'TP':tp,'FP':len(fp),'FN':len(fn)})
            for kind,items in [('FP',fp),('FN',fn)]:
              for item in items:
                rows.append({'model':model,'stage':stage,'class':name,'image_stem':f.stem,'error':kind,**item})
    for filename,data in [('bbox_diagnostic_indicators.csv',rows),('bbox_diagnostic_counts.csv',counts)]:
      with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
    print('Exported',len(rows),'bbox indicators and',len(counts),'counts')
if __name__=='__main__':main()
