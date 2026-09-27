import argparse 
import json 
import os 
import sys 
from pathlib import Path 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
import subprocess 
import time 


BASE_CONFIGS ={
"nwpu":"exps/simplified_ablation/nwpu_full_system.json",
"ucmerced":"exps/simplified_ablation/ucmerced_full_system.json",
"mstar":"exps/simplified_ablation/mstar_full_system.json",
}


def load_json (path :Path ):
    with open (path ,'r',encoding ='utf-8')as f :
        return json .load (f )


def dump_json (obj ,path :Path ):
    path .parent .mkdir (parents =True ,exist_ok =True )
    with open (path ,'w',encoding ='utf-8')as f :
        json .dump (obj ,f ,ensure_ascii =False ,indent =2 )


def param_grids ():
    """Return parameter grids for the retained experiment presets."""
    grids_A ={
    "avg_alpha":[0.30 ,0.60 ,0.75 ,0.85 ,0.95 ],
    "dual_ema_fusion_weight":[0.35 ,0.45 ,0.55 ,0.65 ],
    "EMA_beta":[0.800 ,0.950 ,0.990 ,0.995 ,0.997 ],
    }
    grids_B ={
    "spatial_relation_weight":[0.20 ,0.35 ,0.50 ],
    "modulation_strength":[0.00 ,0.10 ,0.20 ],
    }
    grids_C ={
    "new_class_noise_scale":[0.00 ,0.10 ,0.20 ],
    # anchor_lambda: NWPU defaults ~0.07; provide generic set
    "anchor_lambda":[0.00 ,0.07 ,0.12 ],
    }
    struct_C ={
    "prompt_token_num":[4 ,6 ,8 ],
    "spatial_context_dim":[64 ,128 ,256 ],
    }
    return grids_A ,grids_B ,grids_C ,struct_C 


def build_runs (args ):
    datasets =list (BASE_CONFIGS .keys ())if args .dataset =='all'else [args .dataset ]
    grids_A ,grids_B ,grids_C ,struct_C =param_grids ()

    runs =[]

    for ds in datasets :
    # All datasets supported: nwpu, ucmerced, mstar
        base_path =Path (BASE_CONFIGS [ds ])
        if not base_path .exists ():
            print (f"[WARN] Base config not found: {base_path}")
            continue 
        base_cfg =load_json (base_path )

        # Inject dataset roots via CLI or environment
        if ds =='nwpu':
            nwpu_root =getattr (args ,'nwpu_root',None )or os .environ .get ('NWPU_DATA_ROOT')
            if nwpu_root :
                base_cfg ['nwpu_data_root']=nwpu_root 
        if ds =='ucmerced':
            ucm_root =getattr (args ,'ucm_root',None )or os .environ .get ('UCM_DATA_ROOT')
            if ucm_root :
                base_cfg ['ucm_data_root']=ucm_root 
        if ds =='mstar':
            mstar_root =getattr (args ,'mstar_root',None )or os .environ .get ('MSTAR_DATA_ROOT')
            if mstar_root :
                base_cfg ['mstar_data_root']=mstar_root 

                # enforce single-seed per experiment (seed: [args.seed])
        base_cfg ["seed"]=[args .seed ]

        # category selection
        include_A =(args .category in ['A','all'])
        include_B =(args .category in ['B','all'])
        include_C =(args .category in ['C','all'])

        # A-level (three datasets)
        if include_A :
            for key ,values in grids_A .items ():
                for v in values :
                    cfg =dict (base_cfg )
                    cfg [key ]=v 
                    # tag prefix/model_prefix
                    tag =f"{key}={v}"
                    cfg ["prefix"]=f"{base_cfg.get('prefix','sweep')}_{tag}"
                    cfg ["model_prefix"]=f"{base_cfg.get('model_prefix','sweep')}_{tag}"
                    out_cfg =Path (f"runs/sweeps/{ds}/A/{key}/{v}/config.json")
                    dump_json (cfg ,out_cfg )
                    runs .append ((ds ,key ,v ,out_cfg ))

                    # B-level (three datasets)
        if include_B :
            for key ,values in grids_B .items ():
                for v in values :
                    cfg =dict (base_cfg )
                    cfg [key ]=v 
                    tag =f"{key}={v}"
                    cfg ["prefix"]=f"{base_cfg.get('prefix','sweep')}_{tag}"
                    cfg ["model_prefix"]=f"{base_cfg.get('model_prefix','sweep')}_{tag}"
                    out_cfg =Path (f"runs/sweeps/{ds}/B/{key}/{v}/config.json")
                    dump_json (cfg ,out_cfg )
                    runs .append ((ds ,key ,v ,out_cfg ))

                    # C-level (only NWPU to reduce volume)
        if include_C and ds =='nwpu':
            for key ,values in grids_C .items ():
                for v in values :
                    cfg =dict (base_cfg )
                    cfg [key ]=v 
                    tag =f"{key}={v}"
                    cfg ["prefix"]=f"{base_cfg.get('prefix','sweep')}_{tag}"
                    cfg ["model_prefix"]=f"{base_cfg.get('model_prefix','sweep')}_{tag}"
                    out_cfg =Path (f"runs/sweeps/{ds}/C/{key}/{v}/config.json")
                    dump_json (cfg ,out_cfg )
                    runs .append ((ds ,key ,v ,out_cfg ))

                    # structural params (only NWPU)
            for key ,values in struct_C .items ():
                for v in values :
                    cfg =dict (base_cfg )
                    cfg [key ]=v 
                    tag =f"{key}={v}"
                    cfg ["prefix"]=f"{base_cfg.get('prefix','sweep')}_{tag}"
                    cfg ["model_prefix"]=f"{base_cfg.get('model_prefix','sweep')}_{tag}"
                    out_cfg =Path (f"runs/sweeps/{ds}/C_struct/{key}/{v}/config.json")
                    dump_json (cfg ,out_cfg )
                    runs .append ((ds ,key ,v ,out_cfg ))

                    # Extras (NWPU only)
        if ds =='nwpu'and args .extras :
        # 2x2 interaction: avg_alpha x dual_ema_fusion_weight
            for avg in [0.75 ,0.95 ]:
                for dema in [0.45 ,0.65 ]:
                    cfg =dict (base_cfg )
                    cfg ['avg_alpha']=avg 
                    cfg ['dual_ema_fusion_weight']=dema 
                    tag =f"avg_alpha={avg}__dual_ema_fusion_weight={dema}"
                    cfg ['prefix']=f"{base_cfg.get('prefix','sweep')}_inter_{tag}"
                    cfg ['model_prefix']=f"{base_cfg.get('model_prefix','sweep')}_inter_{tag}"
                    out_cfg =Path (f"runs/sweeps/{ds}/extras/interaction/{tag}/config.json")
                    dump_json (cfg ,out_cfg )
                    runs .append ((ds ,'interaction',tag ,out_cfg ))


    return runs 


def run_job (cfg_path :Path ,python_exec :str =sys .executable ):
    cmd =[python_exec ,'main.py','--config',str (cfg_path )]
    start =time .time ()
    try :
        proc =subprocess .run (cmd ,stdout =subprocess .PIPE ,stderr =subprocess .STDOUT ,text =True ,check =False )
        dur =time .time ()-start 
        return (cfg_path ,proc .returncode ,dur ,proc .stdout )
    except Exception as e :
        return (cfg_path ,-99 ,0.0 ,str (e ))


def main ():
    parser =argparse .ArgumentParser (description ='Sweep TerraSAP experiment parameters')
    parser .add_argument ('--dataset',choices =['nwpu','ucmerced','mstar','all'],default ='all')
    parser .add_argument ('--category',choices =['A','B','C','all'],default ='all')
    parser .add_argument ('--seed',type =int ,default =2025 )
    parser .add_argument ('--max_parallel',type =int ,default =1 )
    parser .add_argument ('--python_exec',type =str ,default =sys .executable ,help ='Python interpreter with PyTorch installed')
    parser .add_argument ('--dry_run',action ='store_true')
    parser .add_argument ('--extras',action ='store_true',help ='Include the NWPU 2x2 EMA-weight interaction grid')
    parser .add_argument ('--nwpu_root',type =str ,default =None ,help ='NWPU-RESISC45 dataset root (contains train/ and test/)')
    parser .add_argument ('--mstar_root',type =str ,default =None ,help ='MSTAR dataset root (contains train/ and test/)')
    parser .add_argument ('--ucm_root',type =str ,default =None ,help ='UCMerced dataset root (contains train/ and test/)')
    args =parser .parse_args ()

    runs =build_runs (args )
    print (f"Prepared {len(runs)} runs.")

    # Write manifest
    manifest =Path ('runs/sweeps/manifest.csv')
    manifest .parent .mkdir (parents =True ,exist_ok =True )
    with open (manifest ,'w',encoding ='utf-8')as f :
        f .write ('dataset,param,value,config_path,command\n')
        for ds ,key ,v ,cfg in runs :
            cmd =f"{sys.executable} main.py --config {cfg}"
            f .write (f"{ds},{key},{v},{cfg},{cmd}\n")
    print (f"Manifest written: {manifest}")

    if args .dry_run or len (runs )==0 :
        print ("Dry run complete. Not executing training runs.")
        return 

        # Preflight: check torch in selected python
    try :
        chk =subprocess .run ([args .python_exec ,'-c','import torch,sys; sys.stdout.write(torch.__version__)'],
        stdout =subprocess .PIPE ,stderr =subprocess .STDOUT ,text =True ,check =False )
        if chk .returncode !=0 or not chk .stdout :
            print ("[ERROR] Selected python cannot import torch. Please set --python_exec to your env with PyTorch.")
            print (chk .stdout )
            return 
        else :
            print (f"Using python: {args.python_exec} (torch {chk.stdout})")
    except Exception as e :
        print (f"[ERROR] Failed to check python env: {e}")
        return 

        # Execute with limited parallelism
    results =[]
    with ThreadPoolExecutor (max_workers =max (1 ,args .max_parallel ))as ex :
        futs =[ex .submit (run_job ,cfg ,args .python_exec )for _ ,_ ,_ ,cfg in runs ]
        for fut in as_completed (futs ):
            results .append (fut .result ())
            cfg_path ,rc ,dur ,_ =results [-1 ]
            status ='OK'if rc ==0 else f'ERR({rc})'
            print (f"[{status}] {cfg_path} in {dur/60:.1f} min")

            # Save a brief summary
    with open ('runs/sweeps/summary.txt','w',encoding ='utf-8')as f :
        for cfg_path ,rc ,dur ,_ in results :
            f .write (f"{cfg_path}, rc={rc}, dur={dur:.1f}s\n")
    print ("Summary saved: runs/sweeps/summary.txt")


if __name__ =='__main__':
    main ()
