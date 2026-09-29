
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import numpy as np
import tensorflow as tf
from PIL import Image,ImageTk
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix

MODEL_PATH=Path("resultados_modelo_A/modelo_barcos_A.keras")
EXT={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
PATTERN=re.compile(r"^([01])__")

class App:
    def __init__(self,root):
        self.root=root; root.title("Detector de barcos - Modelo A"); root.geometry("1050x700")
        if not MODEL_PATH.exists():
            messagebox.showerror("Modelo","Ejecuta primero entrenamiento_modelo_A.py"); root.destroy(); return
        self.model=tf.keras.models.load_model(MODEL_PATH)
        self.files=[]; self.i=0; self.photo=None; self.real=[]; self.pred=[]; self.conf=[]
        self.build()

    def build(self):
        ttk.Label(self.root,text="CLASIFICADOR DE BARCOS - MODELO A",font=("Arial",20,"bold")).pack(pady=12)
        bar=ttk.Frame(self.root); bar.pack(pady=10)
        for text,cmd in [("Seleccionar carpeta",self.folder),("Inferir",self.predict),("Evaluar carpeta",self.evaluate),
                         ("Anterior",self.prev),("Siguiente",self.next)]:
            ttk.Button(bar,text=text,command=cmd).pack(side="left",padx=5)
        body=ttk.Frame(self.root); body.pack(fill="both",expand=True,padx=20)
        lf=ttk.LabelFrame(body,text="Imagen"); lf.pack(side="left",fill="both",expand=True)
        self.img=ttk.Label(lf,text="Seleccione una carpeta"); self.img.pack(fill="both",expand=True)
        rf=ttk.LabelFrame(body,text="Resultado"); rf.pack(side="right",fill="y",padx=15)
        self.vars={k:tk.StringVar(value="-") for k in ["archivo","real","pred","conf"]}
        for k,t in [("archivo","Archivo"),("real","Real"),("pred","Predicción"),("conf","Confianza")]:
            ttk.Label(rf,text=t,font=("Arial",10,"bold")).pack(anchor="w",padx=15,pady=(15,0))
            ttk.Label(rf,textvariable=self.vars[k],font=("Arial",12)).pack(anchor="w",padx=15)
        mf=ttk.LabelFrame(self.root,text="Métricas en vivo"); mf.pack(fill="x",padx=20,pady=15)
        self.mv={k:tk.StringVar(value="-") for k in ["prog","acc","prec","rec","cm"]}
        for c,(k,t) in enumerate([("prog","Progreso"),("acc","Accuracy"),("prec","Precision"),("rec","Recall")]):
            ttk.Label(mf,text=t,font=("Arial",10,"bold")).grid(row=0,column=c,padx=10,pady=5)
            ttk.Label(mf,textvariable=self.mv[k]).grid(row=1,column=c,padx=10,pady=5)
        ttk.Label(mf,text="Matriz [TN FP; FN TP]").grid(row=2,column=0,padx=10,pady=5)
        ttk.Label(mf,textvariable=self.mv["cm"],font=("Consolas",11)).grid(row=2,column=1,columnspan=3)
        self.status=tk.StringVar(value="Listo."); ttk.Label(self.root,textvariable=self.status,relief="sunken").pack(fill="x",side="bottom")

    def folder(self):
        d=filedialog.askdirectory()
        if not d:return
        self.files=sorted([p for p in Path(d).iterdir() if p.suffix.lower() in EXT]); self.i=0
        self.real=[]; self.pred=[]; self.conf=[]
        if not self.files: messagebox.showwarning("Carpeta","No hay imágenes."); return
        self.show(); self.status.set(f"{len(self.files)} imágenes cargadas.")

    def label(self,p):
        m=PATTERN.match(p.name); return int(m.group(1)) if m else None

    def show(self):
        p=self.files[self.i]; im=Image.open(p).convert("RGB"); im.thumbnail((520,500))
        self.photo=ImageTk.PhotoImage(im); self.img.configure(image=self.photo,text="")
        self.vars["archivo"].set(p.name); r=self.label(p); self.vars["real"].set("-" if r is None else ("BARCO" if r else "NO BARCO"))
        self.mv["prog"].set(f"{self.i+1}/{len(self.files)}")

    def predict(self):
        if not self.files:return
        p=self.files[self.i]; im=Image.open(p).convert("RGB").resize((80,80))
        x=np.asarray(im,dtype="float32")/255.; prob=float(self.model.predict(np.expand_dims(x,0),verbose=0)[0][0])
        pred=int(prob>=.5); label="BARCO" if pred else "NO BARCO"; conf=prob if pred else 1-prob
        self.vars["pred"].set(label); self.vars["conf"].set(f"{conf*100:.2f}%")
        r=self.label(p)
        if r is not None:
            self.real.append(r); self.pred.append(pred); self.conf.append(conf); self.metrics()
        self.status.set(f"Inferencia: {p.name}")

    def evaluate(self):
        if not self.files:return
        self.real=[]; self.pred=[]; self.conf=[]
        for i,p in enumerate(self.files):
            r=self.label(p)
            if r is None: continue
            self.i=i; self.show(); self.predict(); self.root.update()
        if not self.real:
            messagebox.showwarning("Etiquetas","Para evaluar automáticamente, los nombres deben empezar por 0__ o 1__.")

    def metrics(self):
        yt=np.array(self.real); yp=np.array(self.pred); cm=confusion_matrix(yt,yp,labels=[0,1])
        self.mv["acc"].set(f"{accuracy_score(yt,yp)*100:.2f}%")
        self.mv["prec"].set(f"{precision_score(yt,yp,zero_division=0)*100:.2f}%")
        self.mv["rec"].set(f"{recall_score(yt,yp,zero_division=0)*100:.2f}%")
        self.mv["cm"].set(f"[[{cm[0,0]} {cm[0,1]}] [{cm[1,0]} {cm[1,1]}]]")

    def next(self):
        if self.files and self.i<len(self.files)-1:self.i+=1; self.show()
    def prev(self):
        if self.files and self.i>0:self.i-=1; self.show()

if __name__=="__main__":
    root=tk.Tk(); App(root); root.mainloop()
