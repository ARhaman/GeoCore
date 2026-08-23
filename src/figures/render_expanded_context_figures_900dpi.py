"""Render print-resolution PNG/TIFF companions for the editable context-figure deck.

The PowerPoint deck remains the editable source. These 900-DPI raster companions are for insertion
into the Word manuscript and use only real images and masks from the mineral study manifest.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAMES = ["background", "olivine", "pyroxene", "plagioclase", "alkali feldspar", "quartz", "biotite", "muscovite", "hornblende"]
PALETTE = [(32,35,40),(80,170,90),(70,125,200),(245,180,65),(190,120,70),(220,220,220),(140,80,155),(225,120,175),(70,175,175)]
NAVY=(18,73,110); TEAL=(26,126,111); BLUE=(50,120,168); RUST=(180,92,72); INK=(20,34,45); PALE=(244,248,250); WHITE=(255,255,255); MID=(75,91,102)
S = 3  # 13.333 × 7.5 in at 900 dpi ÷ 300 layout coordinate scale.
W, H = 4000, 2250  # 13.333 × 7.5 inches at 300 layout units per inch; saved at 900 DPI.


def f(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def frame_number(path):
    match = re.search(r"_(\d+)\.(?:jpg|jpeg|png)$", path.lower())
    return int(match.group(1)) if match else -1


def records_by_source(rows):
    result = defaultdict(list)
    for row in rows: result[row["source_id"]].append(row)
    for vals in result.values(): vals.sort(key=lambda row: frame_number(row["image_path"]))
    return result


def mask_rgb(path):
    with Image.open(path) as src: values = np.asarray(src.convert("L"), dtype=np.uint8)
    arr = np.zeros((*values.shape,3), dtype=np.uint8)
    for i, colour in enumerate(PALETTE): arr[values == i] = colour
    return Image.fromarray(arr, "RGB")


def contain(img, x, y, w, h):
    copy = img.copy(); copy.thumbnail((w,h), Image.Resampling.LANCZOS)
    xx = x+(w-copy.width)//2; yy = y+(h-copy.height)//2
    return copy, xx, yy


def wrapped(draw, value, xy, width, font, fill, align="left", line_gap=5):
    words = value.split(); lines=[]; line=""
    for word in words:
        candidate = (line+" "+word).strip()
        if draw.textlength(candidate, font=font) <= width or not line: line=candidate
        else: lines.append(line); line=word
    if line: lines.append(line)
    x,y=xy
    for text in lines:
        tw=draw.textlength(text,font=font)
        draw.text((x if align=="left" else x+(width-tw)/2,y),text,font=font,fill=fill)
        y += font.size+line_gap
    return y


def rect(draw, xy, fill, outline=None, radius=18):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline or fill, width=5)


def header(draw, number, title, subtitle):
    draw.rectangle((0,0,W,165), fill=NAVY)
    draw.text((110,28), f"Figure {number}. {title}", font=f(77,True), fill=WHITE)
    draw.text((115,185), subtitle, font=f(28), fill=MID)


def panel(canvas, draw, path, label, x, y, w, h, colour):
    rect(draw,(x,y,x+w,y+74),colour,radius=0)
    tw=draw.textlength(label,font=f(25,True));draw.text((x+(w-tw)/2,y+19),label,font=f(25,True),fill=WHITE)
    draw.rectangle((x,y+74,x+w,y+74+h),fill=(25,28,31))
    image = Image.open(path).convert("RGB"); copy,xx,yy=contain(image,x,y+74,w,h);canvas.paste(copy,(xx,yy))


def stage(draw, label, body, x, y, w, colour):
    rect(draw,(x,y,x+w,y+350),PALE,colour)
    draw.rectangle((x,y,x+w,y+80),fill=colour)
    tw=draw.textlength(label,font=f(26,True));draw.text((x+(w-tw)/2,y+25),label,font=f(26,True),fill=WHITE)
    lines=body.split("\n")
    for i,line in enumerate(lines):
        tw=draw.textlength(line,font=f(38,True));draw.text((x+(w-tw)/2,y+135+i*55),line,font=f(38,True),fill=INK)


def save(img, out, stem):
    img.save(out/f"{stem}.png", dpi=(900,900))
    img.save(out/f"{stem}.tif", dpi=(900,900), compression="tiff_lzw")


def figure1(by_source, out):
    src_id="Muscovite-Granite-3" if "Muscovite-Granite-3" in by_source else next(iter(by_source))
    recs=by_source[src_id]; img=Image.new("RGB",(W,H),WHITE);draw=ImageDraw.Draw(img);header(draw,1,"Dataset construction and source-level study design","Original workflow using this study’s registered mineral images, indexed masks, and audited source-level manifest.")
    xs=[95,1050,2005,2960]
    for x,rec in zip(xs[:3],[recs[0],recs[len(recs)//2],recs[-1]]):panel(img,draw,rec["image_path"],f"REAL REGISTERED VIEW {frame_number(rec['image_path'])}",x,300,855,515,TEAL)
    mask_path=out/"_fig1_mask.png";mask_rgb(recs[0]["mask_path"]).save(mask_path);panel(img,draw,mask_path,"ONE SHARED INDEXED MASK",xs[3],300,855,515,RUST)
    wrapped(draw,f"Example source: {src_id}. Eight angular views of the same thin section share registered geometry and one semantic reference mask.",(175,860),3650,f(34,True),INK,"center")
    stages=[("1  ACQUIRE","97 SOURCES\n776 RGB IMAGES",105,1200,BLUE),("2  REGISTER","8 VIEWS/SOURCE\nregistered geometry",865,1200,TEAL),("3  ANNOTATE","1 MASK/SOURCE\n8 mineral classes",1625,1200,RUST),("4  SPLIT SOURCES","72 / 15 / 10\ntrain / val / test",2385,1200,NAVY),("5  VALIDATE","2 × 5 FOLDS\n10 grouped estimates",3145,1200,TEAL)]
    for a,b,x,y,c in stages:stage(draw,a,b,x,y,650,c)
    for i in range(4):draw.line((755+i*760,1375,855+i*760,1375),fill=TEAL,width=10)
    wrapped(draw,"Source-level rule: every registered view from one thin section is kept together in exactly one partition; correlated angular views cannot cross into testing.",(160,1640),3680,f(37,True),NAVY,"center")
    rect(draw,(230,1830,3770,2050),(231,242,244),TEAL);wrapped(draw,"Primary evidence: validation-only model selection followed by repeated grouped source-level cross-validation",(335,1900),3300,f(42,True),INK,"center")
    draw.text((710,2170),"Dataset [1]. This study; high-level workflow organization is original.",font=f(22),fill=MID)
    save(img,out,"Figure_1_dataset_construction_900dpi")


def choose_gallery(by_source):
    info=[]
    for sid,recs in by_source.items():
        with Image.open(recs[0]["mask_path"]) as p: labs=set(int(x) for x in np.unique(np.asarray(p.convert("L"))) if int(x)>0)
        info.append((sid,recs,labs))
    out=[]; covered=set()
    for target in (6,7,8,1,2,3,4,5):
        cand=[q for q in info if target in q[2] and q[0] not in {z[0] for z in out}]
        if cand:
            cand.sort(key=lambda q:(len(q[2]-covered),len(q[2])),reverse=True);out.append(cand[0]);covered|=cand[0][2]
        if len(out)==8:return out
    for q in sorted(info,key=lambda q:len(q[2]-covered),reverse=True):
        if q[0] not in {z[0] for z in out}:out.append(q);covered|=q[2]
        if len(out)==8:break
    return out


def figure2(by_source,out):
    selected=choose_gallery(by_source);img=Image.new("RGB",(W,H),WHITE);draw=ImageDraw.Draw(img);header(draw,2,"Representative registered mineral-image and semantic-mask gallery","Eight real source examples selected to expose mineral, lithological, and optical diversity rather than to optimize appearance.")
    for i,(sid,recs,labs) in enumerate(selected):
        row,col=divmod(i,4);x=95+col*955;y=300+row*795
        panel(img,draw,recs[0]["image_path"],"REAL POLARIZED-LIGHT IMAGE",x,y,850,265,TEAL)
        mask=out/f"_gallery_{i}.png";mask_rgb(recs[0]["mask_path"]).save(mask);panel(img,draw,mask,"INDEXED SEMANTIC MASK",x,y+355,850,265,RUST)
        tw=draw.textlength(sid,font=f(30,True));draw.text((x+(850-tw)/2,y+705),sid,font=f(30,True),fill=INK)
        label=f"{len(labs)} annotated mineral classes";tw=draw.textlength(label,font=f(23));draw.text((x+(850-tw)/2,y+745),label,font=f(23),fill=MID)
    draw.text((130,1950),"MINERAL LEGEND",font=f(27,True),fill=NAVY)
    for i,(name,c) in enumerate(zip(NAMES,PALETTE)):
        column=i%5;row=i//5;x=520+column*655;y=1935+row*65
        draw.rectangle((x,y,x+35,y+35),fill=c);draw.text((x+52,y-2),name,font=f(24,True),fill=INK)
    wrapped(draw,"Each pair retains the original source image and its real semantic mask. Sources were selected to demonstrate observed diversity, not class frequency or model performance.",(500,2110),3000,f(23),MID,"center")
    save(img,out,"Figure_2_representative_mineral_gallery_900dpi")


def simple_box(draw,title,body,x,y,w,colour):stage(draw,title,body,x,y,w,colour)

def figure3(by_source,out):
    rec=(by_source.get("Muscovite-Granite-3") or next(iter(by_source.values())))[0];img=Image.new("RGB",(W,H),WHITE);draw=ImageDraw.Draw(img);header(draw,3,"B7–B8 probability ensemble and validation-only selection","Single-view inference, complementary architectures, probability fusion, and strictly separated model selection.")
    panel(img,draw,rec["image_path"],"REAL INPUT IMAGE",95,360,700,485,TEAL)
    rect(draw,(105,920,785,1050),PALE,TEAL);wrapped(draw,"640 × 512 normalized input  |  448 × 448 training crop",(135,955),620,f(24,True),INK,"center")
    simple_box(draw,"B7  PRETRAINED RESNET-34 U-NET","Weighted cross-entropy\n+ mineral Dice loss",930,390,820,BLUE)
    simple_box(draw,"B8  DEEPLABV3–RESNET50","Weighted cross-entropy\n+ focal Tversky loss",930,1110,820,RUST)
    for y1,y2 in [(650,610),(830,1290)]:draw.line((805,y1,925,y2),fill=TEAL,width=10)
    rect(draw,(1900,700,2850,1210),(231,242,244),TEAL);wrapped(draw,"PROBABILITY-LEVEL ENSEMBLE",(2040,790),680,f(31,True),NAVY,"center");wrapped(draw,"pᵉⁿˢ = w pᴮ⁷ + (1 − w) pᴮ⁸\nŷ = argmax₍c₎ pᵉⁿˢ₍c₎",(2050,930),660,f(43,True),INK,"center")
    simple_box(draw,"VALIDATION-ONLY CALIBRATION","Original split: B7 = 0.6\nB8 = 0.4",3000,390,850,NAVY);simple_box(draw,"GROUPED CV RESELECTION","Fold B7 weights: 0.3–0.6\nmean = 0.44",3000,1110,850,TEAL)
    draw.line((1760,610,1895,900),fill=TEAL,width=10);draw.line((1760,1290,1895,1070),fill=TEAL,width=10);draw.line((2855,950,2995,610),fill=TEAL,width=10);draw.line((2855,1020,2995,1290),fill=TEAL,width=10)
    rect(draw,(195,1680,3800,1915),PALE,RUST);wrapped(draw,"Model-selection safeguard: test sources never selected a checkpoint, crop policy, loss configuration, or ensemble coefficient.",(310,1755),3560,f(38,True),INK,"center")
    draw.text((710,2165),"Mathematical definitions and source-level uncertainty are reported in Section 2.7.",font=f(22),fill=MID)
    save(img,out,"Figure_3_B7_B8_ensemble_framework_900dpi")


def figure4(out):
    img=Image.new("RGB",(W,H),WHITE);draw=ImageDraw.Draw(img);header(draw,4,"Model-development progression and grouped source-level validation","Validated original source-held results, primary repeated grouped-CV estimate, and per-mineral performance.")
    draw.text((120,300),"A. Original fixed source-held test: mean mineral IoU",font=f(39,True),fill=NAVY)
    pro=[("B0 compact U-Net",.0837,BLUE),("B1 class weights",.0908,BLUE),("B2 mineral crops",.1296,TEAL),("B7 ResNet-34",.2138,NAVY),("B8 DeepLabV3",.2096,RUST),("B7–B8 ensemble",.2440,TEAL)]
    for i,(lab,val,col) in enumerate(pro):
        y=470+i*130;draw.text((150,y),lab,font=f(27,True),fill=INK);draw.rectangle((600,y+10,1320,y+44),fill=(222,232,236));draw.rectangle((600,y+10,600+int(720*val/.27),y+44),fill=col);draw.text((1360,y),f"{val:.4f}",font=f(26,True),fill=INK)
    rect(draw,(1620,310,2520,1220),(231,242,244),TEAL);wrapped(draw,"B. PRIMARY GROUPED-CV ESTIMATE",(1690,415),760,f(30,True),NAVY,"center");wrapped(draw,"2 repeats × 5 source-level folds",(1710,555),720,f(31,True),INK,"center");wrapped(draw,"Mineral mIoU\n0.3294 ± 0.0459",(1675,710),330,f(42,True),INK,"center");wrapped(draw,"Mineral Dice\n0.4738 ± 0.0545",(2110,710),330,f(42,True),INK,"center");wrapped(draw,"Approx. 95% mIoU: 0.3009–0.3578\nApprox. 95% Dice: 0.4401–0.5076",(1700,1050),720,f(24),MID,"center")
    draw.text((2670,300),"C. Per-mineral mean IoU across grouped held-out folds",font=f(35,True),fill=NAVY)
    vals=[("Olivine",.3701),("Pyroxene",.2413),("Plagioclase",.2974),("Alkali feldspar",.4035),("Quartz",.3482),("Biotite",.1796),("Muscovite",.4142),("Hornblende",.3807)]
    for i,(lab,val) in enumerate(vals):
        col,row=divmod(i,4);x=2700+col*570;y=520+row*155;c=TEAL if val>=.3 else RUST;draw.rectangle((x,y+10,x+35,y+45),fill=c);draw.text((x+55,y),lab,font=f(26,True),fill=INK);draw.text((x+410,y),f"{val:.3f}",font=f(26,True),fill=INK)
    rect(draw,(160,1450,3840,1800),PALE,NAVY);draw.text((270,1530),"Finding",font=f(33,True),fill=NAVY);wrapped(draw,"The ensemble improved the original source-held test relative to B7 and B8. Cross-validation is the primary performance estimate because it summarizes ten independent source-level hold-out evaluations. Per-class variation is retained rather than concealed by a single macro score.",(590,1505),3050,f(30),INK)
    draw.text((650,2150),"Detailed tables, confusion matrix, class distribution, and fold-specific weights are reported in the manuscript supplement.",font=f(22),fill=MID)
    save(img,out,"Figure_4_grouped_validation_evidence_900dpi")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--manifest",required=True);ap.add_argument("--output",default="artifacts/mineral_segmentation_b7_b8_ensemble/expanded_context_figures_v2/word_900dpi_figures");args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with Path(args.manifest).open(newline="",encoding="utf-8") as h:rows=list(csv.DictReader(h))
    by=records_by_source(rows);figure1(by,out);figure2(by,out);figure3(by,out);figure4(out)
    print(out)

if __name__=="__main__":main()
