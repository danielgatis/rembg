from rembg import remove
from tkinter import Label,Button,filedialog,Tk,Spinbox,BooleanVar,ttk,Frame,IntVar,messagebox, PhotoImage
from threading import Thread
import os, re, json, traceback

# clear log
with open("log.txt","w") as f:
    None
try:

    root = Tk()
    try:
        os.mkdir("output")
    except:
        None
    # define variables  
    var_alpha = BooleanVar()
    var_af = IntVar()
    var_ab = IntVar()
    var_ae = IntVar()
    only_mask = BooleanVar()
    post_process_mask = BooleanVar()
    var_stop = False

    # load setting
    try:
        with open("setting.json", "r") as f:
            setting = json.load(f)
        var_alpha.set(setting["var_alpha"])
        var_af.set(setting["var_af"])
        var_ab.set(setting["var_ab"])
        var_ae.set(setting["var_ae"])
        only_mask.set(setting["only_mask"])
        post_process_mask.set(setting["post_process_mask"])
    except:
        # default setting
        var_alpha.set(False)
        var_af.set(85)
        var_ab.set(130)
        var_ae.set(4)
        only_mask.set(False)
        post_process_mask.set(False)

    # ---------- funtions --------

    def save_setting():
        if not messagebox.askyesno("Save", "Save Settings"):
            return
        setting = {
            "var_alpha":var_alpha.get(),
            "var_af":var_af.get(),
            "var_ab":var_ab.get(),
            "var_ae":var_ae.get(),
            "only_mask":only_mask.get(),
            "post_process_mask":post_process_mask.get()
            }
        setting = json.dumps(setting)
        with open("setting.json", "w") as f:
            f.write(setting)

            
    def funtion(input_path):
        file_name = re.sub(r".+/","",input_path)
        n = re.match(r".+\.", file_name[::-1])
        extension = n.group()[::-1]
        output_path = "output/"+file_name[:-len(extension)]+".png"
        try:
            with open(input_path, 'rb') as i:
                with open(output_path, 'wb') as o:
                    input = i.read()
                    output = remove(
                        input,
                        alpha_matting = var_alpha.get(),
                        alpha_matting_foreground_threshold = int(var_af.get()),
                        alpha_matting_background_threshold = int(var_ab.get()),
                        alpha_matting_erode_size = int(var_ae.get()),
                        only_mask = only_mask.get(),
                        post_process_mask = post_process_mask.get(),
                        )
                    o.write(output)
        except:
            messagebox.showerror("Error",f"error processing image\n{file_name}")
            with open("log.txt","a") as f:
                f.write("\n")
                traceback.print_exc(file=f)



            
        


    def select_path():
        global var_stop
        var_stop = False
        def proceso():
            menu.pack_forget()
            progreso.pack(fill="both")
            
            path = filedialog.askdirectory()
            w = os.walk(top=path)
            lista = []
            for a,b,c in w:
                for file in c:
                    file = file.lower()
                    # check if is image file
                    if ".jpg" in file[-4:] or ".jpeg" in file[-5:] or ".png" in file[-4:]:
                        lista.append(a+"/"+file)
            if lista:
                n=1
                for foto in lista:
                    if var_stop:
                        break
                    progreso["text"]="Processing\n"+str(n)+" Of "+str(len(lista))
                    funtion(foto)
                    n+=1
                    
            menu.pack(fill="both")
            progreso.pack_forget()
        p = Thread(target=proceso)
        p.start()





        
    def select_images():
        global var_stop
        var_stop = False
        def proceso():
            menu.pack_forget()
            progreso.pack(fill="both")
            
            lista = filedialog.askopenfiles(
                filetypes=(
                    "Images *.jpg",
                    "Images *.jpeg",
                    "Images *.png",
                    "Images *.bmp",
                    "All *.*",
                    )
                )
            if lista:
                n=1
                for foto in lista:
                    if var_stop:
                        break
                    progreso["text"]="Processing\n"+str(n)+" Of "+str(len(lista))
                    funtion(foto.name)
                    n+=1

                    
            menu.pack(fill="both")
            progreso.pack_forget()
        p = Thread(target=proceso)
        p.start()




    def bind_chb(a):
        if var_alpha.get():
            cuadro["bg"]="#669999"
            cantidad_af["state"]="disabled"
            cantidad_ab["state"]="disabled"
            cantidad_ae["state"]="disabled"
            texto_af["bg"]="#669999"
            texto_ab["bg"]="#669999"
            texto_ae["bg"]="#669999"
        else:
            cuadro["bg"]="#77dd00"
            cantidad_af["state"]="normal"
            cantidad_ab["state"]="normal"
            cantidad_ae["state"]="normal"
            texto_af["bg"]="#77dd00"
            texto_ab["bg"]="#77dd00"
            texto_ae["bg"]="#77dd00"


    # stop processing images if var_stop is True
    def stop():
        global var_stop
        var_stop = True
        root.destroy()


        
    # ---------- menu principal ------------
    progreso = Label(bg="gray",text="",fg="light blue", font="arial 20",width=18,height=4, relief="groove",borderwidth=10)
    # fondo  
    menu = Frame(bg="gray",width=15,height=250,relief="groove",borderwidth=5)
    menu.pack(fill="both")

    # botones
    Button(
        menu,
        text=" Select Images ",
        command=select_images,
        bg="#0077aa",
        fg="white",
        width=12,
        font=("verdana", 13),
        activebackground="#00ccff",
        relief="raised",
        borderwidth=3
        ).grid(row=0,column=0, sticky="we")

    Button(
        menu,
        text=" Select Folder ",
        command=select_path,
        bg="#0077aa",
        fg="white",
        width=12,
        font=("verdana", 13),
        activebackground="#00ccff",
        relief="raised",
        borderwidth=3
        ).grid(row=0,column=1, sticky="we")


    # chbox alpha buttons container
    f = Frame(menu, relief="raised",borderwidth=3,bg="#669999")
    f.grid(row = 1, columnspan = 2)
    # checkbox alpha
    chb = ttk.Checkbutton(f, text=" Alpha Matting ", var=var_alpha)
    chb.grid(row=0,column=0)
    chb.bind("<ButtonRelease>",bind_chb)
    # alpha options button container
    cuadro = Frame(menu, relief="raised",borderwidth=3,bg="#669999",width=200,height=40)
    cuadro.grid(row = 2, columnspan = 2, sticky="NEWS")

    texto_af = Label(cuadro, text="Foreground threshold", bg="#669999", font=("arial", 12),)
    texto_af.grid(row=0,column=0, padx=6)
    cantidad_af = Spinbox(cuadro, from_=0,to=255, font=("arial", 12),width=6, textvariable=var_af)
    cantidad_af.grid(row = 0, column = 1)

    texto_ab = Label(cuadro, text="Background threshold", bg="#669999", font=("arial", 12),)
    texto_ab.grid(row=1,column=0, padx=6)
    cantidad_ab = Spinbox(cuadro, from_=0,to=255, font=("arial", 12),width=6, textvariable=var_ab)
    cantidad_ab.grid(row = 1, column = 1)

    texto_ae = Label(cuadro, text="Erode size", bg="#669999", font=("arial", 12),)
    texto_ae.grid(row=2,column=0, padx=6)
    cantidad_ae = Spinbox(cuadro, from_=0,to=255, font=("arial", 12),width=6, textvariable=var_ae)
    cantidad_ae.grid(row=2,column=1)
    #------------------------------
    # extra options
    area = Frame(menu, bg="gray")
    area.grid(row = 3, column = 0, columnspan=2)

    chb_mask = ttk.Checkbutton(area, text="Only mask    ", var=only_mask,)
    chb_mask.grid(row = 0, column = 0, padx=3)

    chb_post_mask = ttk.Checkbutton(area, text="Post process mask   ", var=post_process_mask)
    chb_post_mask.grid(row = 0, column = 1, padx=4)

    Button(
        area,
        text="!",
        bg="gray",
        fg="#88eeff",
        bd=0,
        font=("arial" ,16),
        activebackground="gray",
        command=lambda x=0:(
            messagebox.showinfo("About", "Rembg AI v2.0.24 \nRemove photo background\n\nProgramming :\nDaniel Gatis\nUser interface programming:\nErick Esau Martinez")
            )
        ).grid(row = 0, column = 2,padx=6)

    Button(
        area,
        text="Save",
        command=save_setting,
        bg="#669999",
        activebackground="#88bbbb"
        ).grid(row = 0, column = 3, padx=2)




    if not var_alpha.get():
        cuadro["bg"]="#669999"
        cantidad_af["state"]="disabled"
        cantidad_ab["state"]="disabled"
        cantidad_ae["state"]="disabled"
        texto_af["bg"]="#669999"
        texto_ab["bg"]="#669999"
        texto_ae["bg"]="#669999"
    else:
        cuadro["bg"]="#77dd00"
        cantidad_af["state"]="normal"
        cantidad_ab["state"]="normal"
        cantidad_ae["state"]="normal"
        texto_af["bg"]="#77dd00"
        texto_ab["bg"]="#77dd00"
        texto_ae["bg"]="#77dd00"



    root.protocol("WM_DELETE_WINDOW", stop)
    root.geometry("+500+200")
    root.title("Rembg AI")
    root.resizable(0,0)
    root.mainloop()
except:
    with open("log.txt","a") as f:
        f.write("\n")
        traceback.print_exc(file=f)
        raise

        
