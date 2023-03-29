import cv2
import os
import numpy as np
import xlwt



    '''
    img = cv2.imread(filesname+"/"+file)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_binarization = np.where(img_gray > T,255,0).astype(np.uint8)
    img_median = cv2.medianBlur(img_binarization, 5)
    img_median = np.where(img_median > 128,255,0)
    img_gt = cv2.imread("1_median/"+os.path.splitext(file)[0][:5]+"ma"+os.path.splitext(file)[0][10:]+"_binarization_median.jpg",cv2.IMREAD_UNCHANGED)
    img_gt = np.where(img_gt > 128,255,0)
    tp = np.nonzero(np.where(img_median+img_gt > 255,255,0))[0].shape[0]
    fn = np.nonzero(np.where(img_gt-img_median == 255,255,0))[0].shape[0]
    fp = np.nonzero(np.where(img_median-img_gt == 255,255,0))[0].shape[0]
    tn = np.nonzero(np.where(img_median+img_gt == 0,255,0))[0].shape[0]
    bing = tp+fn+fp
    jiao = tp
    iou = jiao/bing
    acc = (tp+tn)/(256*256)
    p = 1 if tp+fp==0 else tp/(tp+fp)
    r = tp/(tp+fn)
    f1 = 0 if p+r==0 else 2*p*r/(p+r)
    hunxiao = np.stack((np.logical_and(img_gt, img_median)*255, img_median, img_gt), axis=-1)
    cv2.imwrite("hunxiao/"+os.path.splitext(file)[0]+"_hunxiao"+str(T)+".jpg",hunxiao)
    return iou,acc,p,r,f1,tp,fn,fp,tn
    
filesname="2"
files = os.listdir(filesname)
files.remove('.DS_Store')
iou = np.zeros([len(files),256])
acc = np.zeros([len(files),256])
p = np.zeros([len(files),256])
r = np.zeros([len(files),256])
f1 = np.zeros([len(files),256])
tp = np.zeros([len(files),256])
fn = np.zeros([len(files),256])
fp = np.zeros([len(files),256])
tn = np.zeros([len(files),256])
for inx,file in enumerate(files):
    print(inx)
    for i in range(0,256,1):
        iou[inx,i],acc[inx,i],p[inx,i],r[inx,i],f1[inx,i],tp[inx,i],fn[inx,i],fp[inx,i],tn[inx,i] = binarization(filesname,file,i)
iou_ave = np.average(iou,0)
acc_ave = np.average(acc,0)
p_ave = np.average(p,0)
r_ave = np.average(r,0)
f1_ave = np.average(f1,0)
tp_ave = np.average(tp,0)
fn_ave = np.average(fn,0)
fp_ave = np.average(fp,0)
tn_ave = np.average(tn,0)
book = xlwt.Workbook()
sheet = book.add_sheet('sheet1')
for i in range(256):
    sheet.write(i,0,iou_ave[i])
    sheet.write(i,1,acc_ave[i])
    sheet.write(i,2,p_ave[i])
    sheet.write(i,3,r_ave[i])
    sheet.write(i,4,f1_ave[i])
    sheet.write(i,5,tp_ave[i])
    sheet.write(i,6,fn_ave[i])
    sheet.write(i,7,fp_ave[i])
    sheet.write(i,8,tn_ave[i])
book.save(r"result1.xls")