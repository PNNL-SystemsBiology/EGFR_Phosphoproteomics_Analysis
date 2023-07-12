## Required Packages and Functions

# %%
import pickle
import sys, os, re
from pathlib import Path
home = str(Path.home())
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [7, 5]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
import venn
import networkx as nx
import copy
from Bio import SeqIO
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")
from plotnine import *
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# from utils import *


#### Useful functions


# %% 
def removeNa(df, colName = "Gene"):
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    if colName in df.columns:
        df = df[df[colName].notna()]
    return(df)
    
# %% 
def removeDu(df):
    du = df.index[df.index.duplicated()]
    dfDu = df.loc[du]
    df = df[~df.index.duplicated(keep='first')]
    for i in du:
        trans = np.log2(np.sum(np.exp2(dfDu.loc[i].values), axis = 0))
        df.loc[i] = trans
    return(df)

# %% 
def corr(A, B, corrMethod="pearson"): # pearson or spearman
    aCols = list(A.columns)
    bCols = list(B.columns)
    aRows = list(A.index)
    bRows = list(B.index)
    intersectCols = list(set(aCols) & set(bCols))
    intersectRows = list(set(aRows) & set(bRows))
    A = A.loc[intersectRows, intersectCols]
    B = B.loc[intersectRows, intersectCols]
    A = A.fillna(0)
    B = B.fillna(0)
    corrList = [A.loc[gene].corr(B.loc[gene], method=corrMethod) for gene in intersectRows]
    correlations = pd.Series(corrList)
    correlations.index = intersectRows
    correlations.name = corrMethod
    return(correlations)

# %% 
def getSubset(df, genes):
    sites = []
    for gene in genes:
        sites += [site for site in df.index if gene in site]
    return(df.loc[sites])
# %% 
def filterCorr(df, thr=0.9):
    dfT = df.transpose()
    corr = dfT.corr()
    links = corr.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']
    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    links_filtered=links.loc[ (links['value'] > thr) & (links['var1'] != links['var2']) ]
    dff = df.loc[set(links_filtered["var1"].tolist() + links_filtered["var2"].tolist()),]
    return(dff)
# %% 
def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

#### Functions for processing phosphopeptides


# %% 
def nip_off_pept(peptide):
    pept_pattern = "\.(.+)\."
    subpept = re.search(pept_pattern, peptide).group(1)
    return(subpept)
# %% 
def strip_peptide(peptide, nip_off = False):
    if nip_off:
        peptide = nip_off_pept(peptide)
    return(re.sub(r"[^A-Za-z]+", '', peptide))
# %% 
def get_ptm_pos_in_pept(peptide, ptm_label = '*', special_chars = r'.]+-=@_!#$%^&*()<>?/\|}{~:['):
    peptide = nip_off_pept(peptide)
    if ptm_label in special_chars:
        ptm_label = '\\' + ptm_label
    ptm_pos = [m.start() for m in re.finditer(ptm_label, peptide)]
    pos = sorted([val - i - 1 for i, val in enumerate(ptm_pos)])
    return(pos)
# %% 
def get_yst(strip_pept, ptm_aa = "YSTyst"):
    return([[i, letter.upper()] for i, letter in enumerate(strip_pept) if letter in ptm_aa])
# %% 
def get_ptm_info(peptide, residue = None, prot_seq = None, ptm_label = '*'):
    if prot_seq != None:
        clean_pept = strip_peptide(peptide, nip_off=True)
        pept_pos = prot_seq.find(clean_pept)
        all_yst = get_yst(clean_pept)
        all_ptm = [[pept_pos + yst[0] + 1, yst[1], yst[0]] for yst in all_yst]
        return(all_ptm)
    if residue != None:
        subpept = nip_off_pept(peptide)
        split_substr = subpept.split(ptm_label)
        res_pos = sorted([int(res) for res in re.findall(r'\d+', residue)])
        first_pos = res_pos[0]
        res_pos.insert(0, first_pos - len(split_substr[0]))
        pept_pos = 0
        all_ptm = []
        for i, res in enumerate(res_pos):
            # print(i)
            if i > 0:
                pept_pos += len(split_substr[i-1])
            yst_pos = get_yst(split_substr[i])
            if len(yst_pos) > 0:
                for j in yst_pos:
                    ptm = [j[0] + res_pos[i] + 1, j[1], pept_pos + j[0]]
                    all_ptm.append(ptm)
        return(all_ptm)
# %% 
def relable_pept(peptide, label_pos, ptm_label = '*'):
    strip_pept = strip_peptide(peptide, nip_off=True)
    for i, pos in enumerate(label_pos):
        strip_pept = strip_pept[:(pos + i + 1)] + ptm_label + strip_pept[(pos + i + 1):]
    return(peptide[:2] + strip_pept + peptide[-2:])
# %% 
def get_phosphositeplus_pos(mod_rsd):
    return([int(re.sub(r"[^0-9]+", '', mod)) for mod in mod_rsd])
def get_mono_pos(mod_rsd):
    return([int(re.sub(r"[^0-9]+", '', mod)) for mod in mod_rsd])
# %% 
def get_res_names(residues):
    res_names = [[res for res in re.findall(r'[A-Z]\d+[a-z\-]+', residue)] if residue[0:2] != 'P_' else [residue] for residue in residues]
    return(res_names)
# %% 
def get_res_pos(residues):
    res_pos = [[int(res) for res in re.findall(r'\d+', residue)] if residue[0:2] != 'P_' else [0] for residue in residues]
    return(res_pos)



# %% 
def reannotate_peptide_wt_seq(phospho_dat, prot_seq, dat_residue_col = "Residue", dat_pept_col = "Peptide", ptm_label = '*'):
    phospho_dat_uniprot = phospho_dat.copy()
    dat_residues = []
    dat_peptides = phospho_dat_uniprot[dat_pept_col].to_list()
    for i, pept in enumerate(dat_peptides):
        clean_pept = strip_peptide(pept, nip_off=True)
        pept_pos = prot_seq.find(clean_pept)
        ptm_pos_in_pept = get_ptm_pos_in_pept(pept, ptm_label)
        ptm_pos = [pept_pos + i + 1 for i in ptm_pos_in_pept]
        new_res = "".join([clean_pept[pos] + str(ptm_pos[i]) + clean_pept[pos].lower() for i, pos in enumerate(ptm_pos_in_pept)])
        dat_residues.append(new_res)
    phospho_dat_uniprot[dat_residue_col] = dat_residues
    return(phospho_dat_uniprot)



# %% 
def old_reannotate_peptide_wt_siteplus(phospho_dat, phosphositeplus, dat_residue_col = "Residue", dat_pept_col = "Peptide", ptm_label = '*', low_thr = 20.0, high_thr = 100.0):
    phosphositeplus_uniprot = phosphositeplus.copy()
    phospho_dat_uniprot = phospho_dat.copy()
    phosphositeplus_pos = get_phosphositeplus_pos(phosphositeplus_uniprot["MOD_RSD"])
    phosphositeplus_evi = np.array([list(phosphositeplus_uniprot["LT_LIT"]), list(phosphositeplus_uniprot["MS_LIT"]), list(phosphositeplus_uniprot["MS_CST"])]).transpose()
    dat_ascore = phospho_dat_uniprot["AScore"].to_list()
    dat_peptides = phospho_dat_uniprot[dat_pept_col].to_list()
    dat_residues = phospho_dat_uniprot[dat_residue_col].to_list()
    dat_pos = get_res_pos(dat_residues)
    for i, residues in enumerate(dat_pos):
        phosphositeplus_check = 0
        for res_pos in residues:
            if res_pos not in phosphositeplus_pos:
                phosphositeplus_check += 1
        if dat_ascore[i] < low_thr or (phosphositeplus_check > 0 and dat_ascore[i] < high_thr) :
            res_num = len(residues)
            ptm_info = get_ptm_info(dat_peptides[i], dat_residues[i], ptm_label)
            possible_ptm_pos = [ptm[0] for ptm in ptm_info]
            possible_ptm_scores = []
            for ptm_pos in possible_ptm_pos:
                if ptm_pos in phosphositeplus_pos:
                    possible_ptm_scores.append(phosphositeplus_evi[phosphositeplus_pos.index(ptm_pos)])
                else:
                    possible_ptm_scores.append(0.0)
            sorted_indice = argsort(possible_ptm_scores)
            new_res_indice = sorted(sorted_indice[-res_num:])
            new_ptm_info = [ptm_info[k] for k in new_res_indice]
            new_residue = ''
            new_res_pos = []
            for new_ptm in new_ptm_info:
                new_residue += new_ptm[1] + str(new_ptm[0]) + new_ptm[1].lower()
                new_res_pos.append(new_ptm[2])
            new_peptide = relable_pept(dat_peptides[i], sorted(new_res_pos), ptm_label)
            dat_peptides[i] = new_peptide
            dat_residues[i] = new_residue
    phospho_dat_uniprot[dat_residue_col] = dat_residues
    phospho_dat_uniprot[dat_pept_col] = dat_peptides
    return(phospho_dat_uniprot)



# %% 
def reannotate_peptide_wt_siteplus(phospho_dat, phosphositeplus, dat_residue_col = "Residue", dat_pept_col = "Peptide", ptm_label = '*', ascore_thr = 60.0, multipho_thr = 60.0):
    phosphositeplus_uniprot = phosphositeplus.copy()
    phospho_dat_uniprot = phospho_dat.copy()
    phosphositeplus_pos = get_phosphositeplus_pos(phosphositeplus_uniprot["MOD_RSD"])
    phosphositeplus_evi = np.array([list(phosphositeplus_uniprot["LT_LIT"]), list(phosphositeplus_uniprot["MS_LIT"]), list(phosphositeplus_uniprot["MS_CST"])]).transpose()
    dat_ascore = phospho_dat_uniprot["AScore"].to_list()
    dat_peptides = phospho_dat_uniprot[dat_pept_col].to_list()
    dat_residues = phospho_dat_uniprot[dat_residue_col].to_list()
    dat_pos = get_res_pos(dat_residues)
    for i, residues in enumerate(dat_pos):
        if dat_ascore[i] < ascore_thr or (dat_ascore[i] < multipho_thr and len(residues) > 1):
            ptm_info = get_ptm_info(dat_peptides[i], dat_residues[i])
            possible_ptm_pos = [ptm[0] for ptm in ptm_info]
            possible_ptm_scores = []
            for ptm_pos in possible_ptm_pos:
                if ptm_pos in phosphositeplus_pos:
                    possible_ptm_scores.append(phosphositeplus_evi[phosphositeplus_pos.index(ptm_pos)])
                else:
                    possible_ptm_scores.append(np.zeros(3))
            possible_ptm_scores = np.nan_to_num(possible_ptm_scores, posinf=0.0, neginf=0.0)
            possible_ptm_scores /= possible_ptm_scores.sum(axis=0, keepdims=True) + 1e-12
            possible_ptm_scores = possible_ptm_scores.mean(axis=1)
            true_res_pos = get_ptm_pos_in_pept(dat_peptides[i], ptm_label)
            ptm_dists = np.array([np.abs(ptm[2] - np.array(true_res_pos)).min() for ptm in ptm_info])
            ptm_dist_scores = 1 - (ptm_dists/ (ptm_dists.max() + 1))
            ptm_final_scores = (ptm_dist_scores + possible_ptm_scores)/2
            possible_ptm_str = '|'.join([ptm[1] + str(ptm[0]) + ptm[1].lower() for ptm in ptm_info])
            num_ptm = len(residues)
            if possible_ptm_scores.sum() == 0.0:
                dat_residues[i] = "".join(['P'] * num_ptm) + '_' + possible_ptm_str
                continue
            new_ptm_indices = np.sort(np.argpartition(ptm_final_scores, -num_ptm)[-num_ptm:])
            new_ptm_info = [ptm_info[ind] for ind in new_ptm_indices]
            new_residue = ''
            new_res_pos = []
            for new_ptm in new_ptm_info:
                new_residue += new_ptm[1] + str(new_ptm[0]) + new_ptm[1].lower()
                new_res_pos.append(new_ptm[2])
            new_peptide = relable_pept(dat_peptides[i], sorted(new_res_pos), ptm_label)
            dat_peptides[i] = new_peptide
            dat_residues[i] = new_residue
    phospho_dat_uniprot[dat_residue_col] = dat_residues
    phospho_dat_uniprot[dat_pept_col] = dat_peptides
    return(phospho_dat_uniprot)



# %% 
def reannotate_peptide_wt_seq_siteplus(phospho_dat, prot_seq, phosphositeplus, dat_residue_col = "Residue", dat_pept_col = "Peptide", ptm_label = '*', ascore_thr = 60.0, multipho_thr = 60.0):
    phosphositeplus_uniprot = phosphositeplus.copy()
    phospho_dat_uniprot = phospho_dat.copy()
    phosphositeplus_pos = get_phosphositeplus_pos(phosphositeplus_uniprot["MOD_RSD"])
    phosphositeplus_evi = np.array([list(phosphositeplus_uniprot["LT_LIT"]), list(phosphositeplus_uniprot["MS_LIT"]), list(phosphositeplus_uniprot["MS_CST"])]).transpose()
    dat_ascore = phospho_dat_uniprot["AScore"].to_list()
    dat_peptides = phospho_dat_uniprot[dat_pept_col].to_list()
    # dat_residues = phospho_dat_uniprot[dat_residue_col].to_list()
    dat_residues = []
    for i, pept in enumerate(dat_peptides):
        clean_pept = strip_peptide(pept, nip_off=True)
        pept_pos = prot_seq.find(clean_pept)
        ptm_pos_in_pept = get_ptm_pos_in_pept(pept, ptm_label)
        ptm_pos = [pept_pos + i + 1 for i in ptm_pos_in_pept]
        new_res = "".join([clean_pept[pos] + str(ptm_pos[i]) + clean_pept[pos].lower() for i, pos in enumerate(ptm_pos_in_pept)])
        dat_residues.append(new_res)
    phospho_dat_uniprot[dat_residue_col] = dat_residues
    dat_pos = get_res_pos(dat_residues)
    for i, residues in enumerate(dat_pos):
        if dat_ascore[i] < ascore_thr or (dat_ascore[i] < multipho_thr and len(residues) > 1):
            ptm_info = get_ptm_info(dat_peptides[i], dat_residues[i])
            possible_ptm_pos = [ptm[0] for ptm in ptm_info]
            possible_ptm_scores = []
            for ptm_pos in possible_ptm_pos:
                if ptm_pos in phosphositeplus_pos:
                    possible_ptm_scores.append(phosphositeplus_evi[phosphositeplus_pos.index(ptm_pos)])
                else:
                    possible_ptm_scores.append(np.zeros(3))
            possible_ptm_scores = np.nan_to_num(possible_ptm_scores, posinf=0.0, neginf=0.0)
            possible_ptm_scores /= possible_ptm_scores.sum(axis=0, keepdims=True) + 1e-12
            possible_ptm_scores = possible_ptm_scores.mean(axis=1)
            true_res_pos = get_ptm_pos_in_pept(dat_peptides[i], ptm_label)
            ptm_dists = np.array([np.abs(ptm[2] - np.array(true_res_pos)).min() for ptm in ptm_info])
            ptm_dist_scores = 1 - (ptm_dists/ (ptm_dists.max() + 1))
            ptm_final_scores = (ptm_dist_scores + possible_ptm_scores)/2
            possible_ptm_str = '|'.join([ptm[1] + str(ptm[0]) + ptm[1].lower() for ptm in ptm_info])
            num_ptm = len(residues)
            if possible_ptm_scores.sum() == 0.0:
                dat_residues[i] = "".join(['P'] * num_ptm) + '_' + possible_ptm_str
                continue
            new_ptm_indices = np.sort(np.argpartition(ptm_final_scores, -num_ptm)[-num_ptm:])
            new_ptm_info = [ptm_info[ind] for ind in new_ptm_indices]
            new_residue = ''
            new_res_pos = []
            for new_ptm in new_ptm_info:
                new_residue += new_ptm[1] + str(new_ptm[0]) + new_ptm[1].lower()
                new_res_pos.append(new_ptm[2])
            new_peptide = relable_pept(dat_peptides[i], sorted(new_res_pos), ptm_label)
            dat_peptides[i] = new_peptide
            dat_residues[i] = new_residue
    phospho_dat_uniprot[dat_residue_col] = dat_residues
    phospho_dat_uniprot[dat_pept_col] = dat_peptides
    return(phospho_dat_uniprot)



# %% 
def reannotate_peptide(phospho_dat, phosphositeplus, prot_seq = None, dat_residue_col = "Residue", dat_pept_col = "Peptide", ptm_label = '*', ascore_thr = 60.0, multipho_thr = 60.0, MS_W = 1.0, LT_W = 1.0, MP_W = 1.0, DT_W = 1.0):
    phosphositeplus_uniprot = phosphositeplus.copy()
    phospho_dat_uniprot = phospho_dat.copy()
    phosphositeplus_pos = get_phosphositeplus_pos(phosphositeplus_uniprot["MOD_RSD"])
    dat_ascore = phospho_dat_uniprot["AScore"].to_list()
    dat_peptides = phospho_dat_uniprot[dat_pept_col].to_list()
    if not prot_seq:
        dat_residues = phospho_dat_uniprot[dat_residue_col].to_list()
    else:
        dat_residues = []
        for i, pept in enumerate(dat_peptides):
            clean_pept = strip_peptide(pept, nip_off=True)
            pept_pos = prot_seq.find(clean_pept)
            ptm_pos_in_pept = get_ptm_pos_in_pept(pept, ptm_label)
            ptm_pos = [pept_pos + i + 1 for i in ptm_pos_in_pept]
            new_res = "".join([clean_pept[pos] + str(ptm_pos[i]) + clean_pept[pos].lower() for i, pos in enumerate(ptm_pos_in_pept)]) 
            dat_residues.append(new_res)
    phospho_dat_uniprot[dat_residue_col] = dat_residues
    dat_pos = get_res_pos(dat_residues)
    phospho_dat["PhosphoNum"] = [len(res) for res in dat_pos]
    # phospho_dat_uniprot_good = phospho_dat[(phospho_dat["AScore"] > ascore_thr) & (phospho_dat["PhosphoNum"] == 1)].copy()
    phospho_dat_uniprot_good = phospho_dat[(phospho_dat["AScore"] > ascore_thr) & (phospho_dat["PhosphoNum"] == 1)].copy()
    good_poses = get_phosphositeplus_pos(phospho_dat_uniprot_good[dat_residue_col])
    phosphositeplus_this = [1 if i in good_poses else 0 for i in phosphositeplus_pos]
    phosphositeplus_evi = np.array([list(phosphositeplus_uniprot["MS_CST"].fillna(0) + phosphositeplus_uniprot["MS_LIT"].fillna(0)), list(phosphositeplus_uniprot["LT_LIT"].fillna(0))]).transpose()
    for i, residues in enumerate(dat_pos):
        if dat_ascore[i] < ascore_thr or (dat_ascore[i] < multipho_thr and len(residues) > 1):
            ptm_info = get_ptm_info(dat_peptides[i], dat_residues[i])
            possible_ptm_pos = [ptm[0] for ptm in ptm_info]
            possible_ptm_scores = []
            monopho_ptm_scores = []
            for ptm_pos in possible_ptm_pos:
                if ptm_pos in phosphositeplus_pos:
                    possible_ptm_scores.append(phosphositeplus_evi[phosphositeplus_pos.index(ptm_pos)])
                    monopho_ptm_scores.append(phosphositeplus_this[phosphositeplus_pos.index(ptm_pos)])
                elif ptm_pos in good_poses:
                    possible_ptm_scores.append([0.0] * 2)
                    monopho_ptm_scores.append(1.0)
                else:
                    possible_ptm_scores.append([0.0] * 2)
                    monopho_ptm_scores.append(0.0)
            possible_ptm_scores /= (np.array(possible_ptm_scores).sum(axis=0, keepdims=True) + 1e-12)
            monopho_ptm_scores = np.array(monopho_ptm_scores)
            true_res_pos = get_ptm_pos_in_pept(dat_peptides[i], ptm_label)
            ptm_dists = np.array([np.abs(ptm[2] - np.array(true_res_pos)).min() for ptm in ptm_info])
            ptm_dist_scores = dat_ascore[i] * (1 - (ptm_dists / (ptm_dists.max() + 1)))
            ptm_dist_scores /= (ptm_dist_scores.sum() + 1e-12)
            ptm_final_scores = possible_ptm_scores[:,0] * MS_W + possible_ptm_scores[:,1] * LT_W + monopho_ptm_scores * MP_W + ptm_dist_scores * DT_W
            possible_ptm_str = '|'.join([ptm[1] + str(ptm[0]) + ptm[1].lower() for ptm in ptm_info])
            num_ptm = len(residues)
            if ptm_final_scores.sum() == 0.0:
                dat_residues[i] = "".join(['P'] * num_ptm) + '_' + possible_ptm_str
                continue
            new_ptm_indices = np.sort(np.argpartition(ptm_final_scores, -num_ptm)[-num_ptm:])
            new_ptm_info = [ptm_info[ind] for ind in new_ptm_indices]
            new_residue = ''
            new_res_pos = []
            for new_ptm in new_ptm_info:
                new_residue += new_ptm[1] + str(new_ptm[0]) + new_ptm[1].lower()
                new_res_pos.append(new_ptm[2])
            new_peptide = relable_pept(dat_peptides[i], sorted(new_res_pos), ptm_label)
            dat_peptides[i] = new_peptide
            dat_residues[i] = new_residue
    phospho_dat_uniprot[dat_residue_col] = dat_residues
    phospho_dat_uniprot[dat_pept_col] = dat_peptides
    return(phospho_dat_uniprot)


##### Get cluster information from phosphositeplus data


# %% 
def get_cluster_info(res_pos, phosphositeplus):
    phosphositeplus_pos = phosphositeplus["Residual Position"].to_list()
    phosphositeplus_cluster = phosphositeplus["Cluster in Protein"].to_list()
    cluster_list = [','.join([str(phosphositeplus_cluster[phosphositeplus_pos.index(i)]) if i in phosphositeplus_pos else str(0) for i in res]) for res in res_pos]
    return(cluster_list)


##### Reannotate the peptides then rolling up to site and cluster levels


# %% 
def reannotate_and_3d_cluster(phospho_peptide, phosphositeplus_clustered, proteome_seqs, val_cols, out_name = "with_reassignment", avgRII_col = "avgRII", dat_uniprot_col = "UniProt", dat_residue_col = "Residue", dat_pept_col = "Peptide", dat_site_col = "SiteID", dat_prot_col = "Protein", ascore_col = "AScore", ptm_label = '*', ascore_thr = 60.0, multipho_thr = 60.0, MS_W = 1.0, LT_W = 1.0, MP_W = 1.0, DT_W = 1.0):
    uniprot_ids = list(set(phospho_peptide[dat_uniprot_col]))
    new_phospho_dat_rean_list = []
    new_phospho_dat_pept_list = []
    new_phospho_dat_site_list = []
    new_phospho_dat_clus_list = []
    new_phospho_dat_single_site_list = []
    for uniprot_id in uniprot_ids:
        phosphositeplus_uniprot = phosphositeplus_clustered[phosphositeplus_clustered["ACC_ID"] == uniprot_id].copy()
        phospho_dat_uniprot = phospho_peptide[phospho_peptide[dat_uniprot_col] == uniprot_id].copy()
        if uniprot_id in proteome_seqs.keys():
            prot_seq_uniprot = proteome_seqs[uniprot_id]
        else:
            prot_seq_uniprot = None
        new_phospho_dat_pept_uniprot_reannot = reannotate_peptide(phospho_dat_uniprot, phosphositeplus_uniprot, prot_seq = prot_seq_uniprot, dat_residue_col = dat_residue_col, dat_pept_col = dat_pept_col, ptm_label = ptm_label, ascore_thr = ascore_thr, multipho_thr = multipho_thr, MS_W = MS_W, LT_W = LT_W, MP_W = MP_W, DT_W = DT_W)
        new_phospho_dat_pept_uniprot = new_phospho_dat_pept_uniprot_reannot.copy()
        new_phospho_dat_rean_list.append(new_phospho_dat_pept_uniprot_reannot)
        new_phospho_dat_pept_uniprot[dat_site_col] = new_phospho_dat_pept_uniprot[dat_prot_col] + '-' + new_phospho_dat_pept_uniprot[dat_residue_col]
        new_phospho_dat_pept_uniprot[val_cols] = 2 ** new_phospho_dat_pept_uniprot[val_cols].add(new_phospho_dat_pept_uniprot[avgRII_col], axis = 0)
        new_phospho_dat_pept_uniprot[val_cols] = new_phospho_dat_pept_uniprot[val_cols].fillna(0)
        # Rolling up to the peptide level
        new_phospho_dat_pept_uniprot["id"] = new_phospho_dat_pept_uniprot[dat_prot_col] + '-' + new_phospho_dat_pept_uniprot[dat_pept_col]
        agg_methods_0 = {
            dat_pept_col : lambda x: x.iloc[0] ,
            dat_site_col : lambda x: x.iloc[0] ,
            dat_uniprot_col : lambda x: x.iloc[0] ,
            dat_prot_col : lambda x: x.iloc[0] ,
            dat_residue_col : lambda x: x.iloc[0] ,
            avgRII_col : lambda x: x.iloc[0] ,
            ascore_col : "min" 
        }
        agg_methods_1 = {i : "sum" for i in val_cols} 
        new_phospho_dat_pept_uniprot = new_phospho_dat_pept_uniprot.groupby("id", as_index=False).agg({**agg_methods_0, **agg_methods_1})
        new_phospho_dat_site_uniprot = new_phospho_dat_pept_uniprot.copy()
        new_phospho_dat_pept_uniprot[val_cols] = np.log2(new_phospho_dat_pept_uniprot[val_cols])
        new_phospho_dat_pept_uniprot[val_cols] = new_phospho_dat_pept_uniprot[val_cols].replace([np.inf, -np.inf], np.nan)
        new_phospho_dat_pept_uniprot[avgRII_col] = np.nanmean(new_phospho_dat_pept_uniprot[val_cols], axis = 1)
        new_phospho_dat_pept_uniprot[val_cols] = new_phospho_dat_pept_uniprot[val_cols].sub(new_phospho_dat_pept_uniprot[avgRII_col], axis=0)
        new_phospho_dat_pept_list.append(new_phospho_dat_pept_uniprot)
        # Rolling up to site leve
        new_pept_col = "All_" + dat_pept_col
        new_phospho_dat_site_uniprot["id"] = new_phospho_dat_site_uniprot[dat_prot_col] + '-' + new_phospho_dat_site_uniprot[dat_residue_col]
        new_phospho_dat_site_uniprot[new_pept_col] = new_phospho_dat_site_uniprot[dat_pept_col] + ' (' + ascore_col + ': ' + new_phospho_dat_site_uniprot[ascore_col].astype(str) + ', ' + avgRII_col + ': ' + new_phospho_dat_site_uniprot[avgRII_col].astype(str) + ')'
        agg_methods_0 = {
            new_pept_col : lambda x: '; '.join(x) ,
            dat_site_col : lambda x: x.iloc[0] ,
            dat_uniprot_col : lambda x: x.iloc[0] ,
            dat_prot_col : lambda x: x.iloc[0] ,
            dat_residue_col : lambda x: x.iloc[0] ,
            avgRII_col : lambda x: x.iloc[0] ,
            ascore_col : "min" 
        }
        new_phospho_dat_site_uniprot = new_phospho_dat_site_uniprot.groupby("id", as_index=False).agg({**agg_methods_0, **agg_methods_1})
        cluster_col = "Cluster"
        new_phospho_dat_site_uniprot[cluster_col] = get_cluster_info(get_res_pos(new_phospho_dat_site_uniprot[dat_residue_col]), phosphositeplus_uniprot) 
        new_phospho_dat_single_site_uniprot = new_phospho_dat_site_uniprot.copy()
        new_phospho_dat_site_uniprot[val_cols] = np.log2(new_phospho_dat_site_uniprot[val_cols])
        new_phospho_dat_site_uniprot[val_cols] = new_phospho_dat_site_uniprot[val_cols].replace([np.inf, -np.inf], np.nan)
        new_phospho_dat_site_uniprot[avgRII_col] = np.nanmean(new_phospho_dat_site_uniprot[val_cols], axis = 1)
        new_phospho_dat_site_uniprot[val_cols] = new_phospho_dat_site_uniprot[val_cols].sub(new_phospho_dat_site_uniprot[avgRII_col], axis=0)
        new_phospho_dat_site_list.append(new_phospho_dat_site_uniprot)
        # Splitting into single site then roll up to single site level
        new_phospho_dat_single_site_uniprot[dat_residue_col] = [','.join(res) for res in get_res_names(new_phospho_dat_single_site_uniprot[dat_residue_col])]
        temp_split = new_phospho_dat_single_site_uniprot[dat_residue_col]
        temp_split = new_phospho_dat_single_site_uniprot[dat_residue_col].str.split(',').apply(pd.Series, 1).stack()

        temp_split.index = temp_split.index.droplevel(-1)
        temp_split.name = dat_residue_col
        new_phospho_dat_single_site_uniprot = new_phospho_dat_single_site_uniprot.drop(dat_residue_col, axis=1)
        new_phospho_dat_single_site_uniprot = new_phospho_dat_single_site_uniprot.join(temp_split)
        new_phospho_dat_single_site_uniprot["id"] = new_phospho_dat_single_site_uniprot[dat_prot_col] + '-' + new_phospho_dat_single_site_uniprot[dat_residue_col]
        agg_methods_0 = {
            new_pept_col : lambda x: '; '.join(x) ,
            dat_site_col : lambda x: x.iloc[0] ,
            dat_uniprot_col : lambda x: x.iloc[0] ,
            dat_prot_col : lambda x: x.iloc[0] ,
            dat_residue_col : lambda x: x.iloc[0] ,
            cluster_col : lambda x: x.iloc[0] ,
            avgRII_col : lambda x: x.iloc[0] ,
            ascore_col : "min" 
        }  
        new_phospho_dat_single_site_uniprot = new_phospho_dat_single_site_uniprot.groupby("id", as_index=False).agg({**agg_methods_0, **agg_methods_1})
        new_phospho_dat_single_site_uniprot[dat_site_col] = new_phospho_dat_single_site_uniprot[dat_prot_col] + '-' + new_phospho_dat_single_site_uniprot[dat_residue_col]
        new_phospho_dat_single_site_uniprot[cluster_col] = get_cluster_info(get_res_pos(new_phospho_dat_single_site_uniprot[dat_residue_col]), phosphositeplus_uniprot) 
        new_phospho_dat_clus_uniprot = new_phospho_dat_single_site_uniprot.copy()
        new_phospho_dat_single_site_uniprot[val_cols] = np.log2(new_phospho_dat_single_site_uniprot[val_cols])
        new_phospho_dat_single_site_uniprot[val_cols] = new_phospho_dat_single_site_uniprot[val_cols].replace([np.inf, -np.inf], np.nan)
        new_phospho_dat_single_site_uniprot[avgRII_col] = np.nanmean(new_phospho_dat_single_site_uniprot[val_cols], axis = 1)
        new_phospho_dat_single_site_uniprot[val_cols] = new_phospho_dat_single_site_uniprot[val_cols].sub(new_phospho_dat_single_site_uniprot[avgRII_col], axis=0)
        new_phospho_dat_single_site_list.append(new_phospho_dat_single_site_uniprot)
        # Rolling up to the cluster level
        new_phospho_dat_clus_uniprot["id"] = new_phospho_dat_clus_uniprot[dat_prot_col] + '-' + new_phospho_dat_clus_uniprot[cluster_col].astype(str)
        agg_methods_0 = {
            new_pept_col : lambda x: '; '.join(x) ,
            dat_site_col : lambda x: x.iloc[0] ,
            dat_uniprot_col : lambda x: x.iloc[0] ,
            dat_prot_col : lambda x: x.iloc[0] ,
            dat_residue_col : lambda x: ''.join(x) ,
            cluster_col : lambda x: x.iloc[0] ,
            avgRII_col : lambda x: x.iloc[0] ,
            ascore_col : "min" 
        }  
        new_phospho_dat_clus_uniprot = new_phospho_dat_clus_uniprot.groupby("id", as_index=False).agg({**agg_methods_0, **agg_methods_1})
        new_phospho_dat_clus_uniprot[val_cols] = np.log2(new_phospho_dat_clus_uniprot[val_cols])
        new_phospho_dat_clus_uniprot[val_cols] = new_phospho_dat_clus_uniprot[val_cols].replace([np.inf, -np.inf], np.nan)
        new_phospho_dat_clus_uniprot[avgRII_col] = np.nanmean(new_phospho_dat_clus_uniprot[val_cols], axis = 1)
        new_phospho_dat_clus_uniprot[val_cols] = new_phospho_dat_clus_uniprot[val_cols].sub(new_phospho_dat_clus_uniprot[avgRII_col], axis=0)
        new_phospho_dat_clus_uniprot[dat_site_col] = new_phospho_dat_clus_uniprot[dat_prot_col] + '-' + new_phospho_dat_clus_uniprot[dat_residue_col]
        new_phospho_dat_clus_uniprot["id"] = new_phospho_dat_clus_uniprot[dat_site_col]
        new_phospho_dat_clus_list.append(new_phospho_dat_clus_uniprot)
    new_phospho_dat_rean = pd.concat(new_phospho_dat_rean_list)
    new_phospho_dat_pept = pd.concat(new_phospho_dat_pept_list)
    new_phospho_dat_site = pd.concat(new_phospho_dat_site_list)
    new_phospho_dat_single_site = pd.concat(new_phospho_dat_single_site_list)
    new_phospho_dat_clus = pd.concat(new_phospho_dat_clus_list)
    new_phospho_dat_rean.to_csv(out_name + "_reannotated_peptide.csv", index=False)
    new_phospho_dat_rean.to_pickle(out_name + "_reannotated_peptide.pkl")
    new_phospho_dat_pept.to_csv(out_name + "_unique_peptide.csv", index=False)
    new_phospho_dat_pept.to_pickle(out_name + "_unique_peptide.pkl")
    new_phospho_dat_site.to_csv(out_name + "_site.csv", index=False)
    new_phospho_dat_site.to_pickle(out_name + "_site.pkl")
    new_phospho_dat_single_site.to_csv(out_name + "_single_site.csv", index=False)
    new_phospho_dat_single_site.to_pickle(out_name + "_single_site.pkl")
    new_phospho_dat_clus.to_csv(out_name + "_cluster.csv", index=False)
    new_phospho_dat_clus.to_pickle(out_name + "_cluster.pkl")
    # return([new_phospho_dat_rean, new_phospho_dat_pept, new_phospho_dat_site, new_phospho_dat_single_site, new_phospho_dat_clus])


# %% 
def reannotate_single_site(phospho_peptide, phosphositeplus_clustered, proteome_seqs, val_cols, out_name = "with_reassignment", avgRII_col = "avgRII", dat_uniprot_col = "UniProt", dat_residue_col = "Residue", dat_pept_col = "Peptide", dat_site_col = "SiteID", dat_prot_col = "Protein", ascore_col = "AScore", ptm_label = '*', ascore_thr = 60.0, multipho_thr = 60.0, MS_W = 1.0, LT_W = 1.0, MP_W = 1.0, DT_W = 1.0):
    uniprot_ids = list(set(phospho_peptide[dat_uniprot_col]))
    new_phospho_dat_single_site_list = []
    for uniprot_id in uniprot_ids:
        phosphositeplus_uniprot = phosphositeplus_clustered[phosphositeplus_clustered["ACC_ID"] == uniprot_id].copy()
        phospho_dat_uniprot = phospho_peptide[phospho_peptide[dat_uniprot_col] == uniprot_id].copy()
        if uniprot_id in proteome_seqs.keys():
            prot_seq_uniprot = proteome_seqs[uniprot_id]
        else:
            prot_seq_uniprot = None
        new_phospho_dat_pept_uniprot_reannot = reannotate_peptide(phospho_dat_uniprot, phosphositeplus_uniprot, prot_seq = prot_seq_uniprot, dat_residue_col = dat_residue_col, dat_pept_col = dat_pept_col, ptm_label = ptm_label, ascore_thr = ascore_thr, multipho_thr = multipho_thr, MS_W = MS_W, LT_W = LT_W, MP_W = MP_W, DT_W = DT_W)
        new_phospho_dat_pept_uniprot = new_phospho_dat_pept_uniprot_reannot.copy()
        new_phospho_dat_pept_uniprot[dat_site_col] = new_phospho_dat_pept_uniprot[dat_prot_col] + '-' + new_phospho_dat_pept_uniprot[dat_residue_col]
        new_phospho_dat_pept_uniprot[val_cols] = 2 ** new_phospho_dat_pept_uniprot[val_cols].add(new_phospho_dat_pept_uniprot[avgRII_col], axis = 0)
        new_phospho_dat_pept_uniprot[val_cols] = new_phospho_dat_pept_uniprot[val_cols].fillna(0)
        # Rolling up to the peptide level
        new_phospho_dat_pept_uniprot["id"] = new_phospho_dat_pept_uniprot[dat_prot_col] + '-' + new_phospho_dat_pept_uniprot[dat_pept_col]
        agg_methods_0 = {
            dat_pept_col : lambda x: x.iloc[0] ,
            dat_site_col : lambda x: x.iloc[0] ,
            dat_uniprot_col : lambda x: x.iloc[0] ,
            dat_prot_col : lambda x: x.iloc[0] ,
            dat_residue_col : lambda x: x.iloc[0] ,
            avgRII_col : lambda x: x.iloc[0] ,
            ascore_col : "min" 
        }
        agg_methods_1 = {i : "sum" for i in val_cols} 
        new_phospho_dat_pept_uniprot = new_phospho_dat_pept_uniprot.groupby("id", as_index=False).agg({**agg_methods_0, **agg_methods_1})
        new_phospho_dat_site_uniprot = new_phospho_dat_pept_uniprot.copy()
        new_phospho_dat_pept_uniprot[val_cols] = np.log2(new_phospho_dat_pept_uniprot[val_cols])
        new_phospho_dat_pept_uniprot[val_cols] = new_phospho_dat_pept_uniprot[val_cols].replace([np.inf, -np.inf], np.nan)
        new_phospho_dat_pept_uniprot[avgRII_col] = np.nanmean(new_phospho_dat_pept_uniprot[val_cols], axis = 1)
        new_phospho_dat_pept_uniprot[val_cols] = new_phospho_dat_pept_uniprot[val_cols].sub(new_phospho_dat_pept_uniprot[avgRII_col], axis=0)
        # Rolling up to site leve
        new_pept_col = "All_" + dat_pept_col
        new_phospho_dat_site_uniprot["id"] = new_phospho_dat_site_uniprot[dat_prot_col] + '-' + new_phospho_dat_site_uniprot[dat_residue_col]
        new_phospho_dat_site_uniprot[new_pept_col] = new_phospho_dat_site_uniprot[dat_pept_col] + ' (' + ascore_col + ': ' + new_phospho_dat_site_uniprot[ascore_col].astype(str) + ', ' + avgRII_col + ': ' + new_phospho_dat_site_uniprot[avgRII_col].astype(str) + ')'
        agg_methods_0 = {
            new_pept_col : lambda x: '; '.join(x) ,
            dat_site_col : lambda x: x.iloc[0] ,
            dat_uniprot_col : lambda x: x.iloc[0] ,
            dat_prot_col : lambda x: x.iloc[0] ,
            dat_residue_col : lambda x: x.iloc[0] ,
            avgRII_col : lambda x: x.iloc[0] ,
            ascore_col : "min" 
        }
        new_phospho_dat_site_uniprot = new_phospho_dat_site_uniprot.groupby("id", as_index=False).agg({**agg_methods_0, **agg_methods_1})
        cluster_col = "Cluster"
        new_phospho_dat_site_uniprot[cluster_col] = get_cluster_info(get_res_pos(new_phospho_dat_site_uniprot[dat_residue_col]), phosphositeplus_uniprot) 
        new_phospho_dat_single_site_uniprot = new_phospho_dat_site_uniprot.copy()
        new_phospho_dat_site_uniprot[val_cols] = np.log2(new_phospho_dat_site_uniprot[val_cols])
        new_phospho_dat_site_uniprot[val_cols] = new_phospho_dat_site_uniprot[val_cols].replace([np.inf, -np.inf], np.nan)
        new_phospho_dat_site_uniprot[avgRII_col] = np.nanmean(new_phospho_dat_site_uniprot[val_cols], axis = 1)
        new_phospho_dat_site_uniprot[val_cols] = new_phospho_dat_site_uniprot[val_cols].sub(new_phospho_dat_site_uniprot[avgRII_col], axis=0)
        # Splitting into single site then roll up to single site level
        new_phospho_dat_single_site_uniprot[dat_residue_col] = [','.join(res) for res in get_res_names(new_phospho_dat_single_site_uniprot[dat_residue_col])]
        temp_split = new_phospho_dat_single_site_uniprot[dat_residue_col]
        temp_split = new_phospho_dat_single_site_uniprot[dat_residue_col].str.split(',').apply(pd.Series, 1).stack()

        temp_split.index = temp_split.index.droplevel(-1)
        temp_split.name = dat_residue_col
        new_phospho_dat_single_site_uniprot = new_phospho_dat_single_site_uniprot.drop(dat_residue_col, axis=1)
        new_phospho_dat_single_site_uniprot = new_phospho_dat_single_site_uniprot.join(temp_split)
        new_phospho_dat_single_site_uniprot["id"] = new_phospho_dat_single_site_uniprot[dat_prot_col] + '-' + new_phospho_dat_single_site_uniprot[dat_residue_col]
        agg_methods_0 = {
            new_pept_col : lambda x: '; '.join(x) ,
            dat_site_col : lambda x: x.iloc[0] ,
            dat_uniprot_col : lambda x: x.iloc[0] ,
            dat_prot_col : lambda x: x.iloc[0] ,
            dat_residue_col : lambda x: x.iloc[0] ,
            cluster_col : lambda x: x.iloc[0] ,
            avgRII_col : lambda x: x.iloc[0] ,
            ascore_col : "min" 
        }  
        new_phospho_dat_single_site_uniprot = new_phospho_dat_single_site_uniprot.groupby("id", as_index=False).agg({**agg_methods_0, **agg_methods_1})
        new_phospho_dat_single_site_uniprot[dat_site_col] = new_phospho_dat_single_site_uniprot[dat_prot_col] + '-' + new_phospho_dat_single_site_uniprot[dat_residue_col]
        new_phospho_dat_single_site_uniprot[cluster_col] = get_cluster_info(get_res_pos(new_phospho_dat_single_site_uniprot[dat_residue_col]), phosphositeplus_uniprot) 
        new_phospho_dat_single_site_uniprot[val_cols] = np.log2(new_phospho_dat_single_site_uniprot[val_cols])
        new_phospho_dat_single_site_uniprot[val_cols] = new_phospho_dat_single_site_uniprot[val_cols].replace([np.inf, -np.inf], np.nan)
        new_phospho_dat_single_site_uniprot[avgRII_col] = np.nanmean(new_phospho_dat_single_site_uniprot[val_cols], axis = 1)
        new_phospho_dat_single_site_uniprot[val_cols] = new_phospho_dat_single_site_uniprot[val_cols].sub(new_phospho_dat_single_site_uniprot[avgRII_col], axis=0)
        new_phospho_dat_single_site_list.append(new_phospho_dat_single_site_uniprot)
    new_phospho_dat_single_site = pd.concat(new_phospho_dat_single_site_list)
    return(new_phospho_dat_single_site)




# %% 
def remove_uncertain_sites(df, fc_cols, icol = None):
    if icol is not None:
        df.index = df[icol]
    new_index = [site for site in df.index.to_list() if not re.search("-[Pp]+_", site)]
    return(df.loc[new_index, fc_cols].copy())



# %% 
def remove_uncertain_rows(df, icol = None):
    if icol is not None:
        df.index = df[icol]
    new_index = [site for site in df.index.to_list() if not re.search("-[Pp]+_", site)]
    return(df.loc[new_index].copy())


### Calculating the fold changes

# %% 
def calculate_fc_ts(uniprot_prot, val_cols = None, time_cols = ["0 min", "2 min", "4 min", "8 min", "12 min"]):
    pho_FC = uniprot_prot.copy()
    for i, t in enumerate(time_cols):
        pho_FC[t + "_FC_1"] = pho_FC[val_cols[(i)*2]] - pho_FC[val_cols[0]]
        pho_FC[t + "_FC_2"] = pho_FC[val_cols[(i)*2+1]] - pho_FC[val_cols[1]]
        pho_FC[t + "_FC_3"] = pho_FC[val_cols[(i)*2]] - pho_FC[val_cols[0]]
        pho_FC[t + "_FC_4"] = pho_FC[val_cols[(i)*2+1]] - pho_FC[val_cols[1]]
        pho_FC[t + "_FC_mean"] = pho_FC[[t + "_FC_1", t + "_FC_2", t + "_FC_3", t + "_FC_4"]].mean(axis=1)
        pho_FC[t + "_FC_std"] = pho_FC[[t + "_FC_1", t + "_FC_2", t + "_FC_3", t + "_FC_4"]].std(axis=1)
    pho_FC = pho_FC.replace([np.inf, -np.inf], np.nan)
    return(pho_FC)
# %% 
def calculate_fc_ts_old(uniprot_prot1, uniprot_prot2, time_cols = ["0 min", "2 min", "4 min", "8 min", "12 min"]):
    # pho_FC = pd.DataFrame(columns = ["MCF10A at " + str(i) + " min" for i in timepoints])
    pho_FC = pd.DataFrame().reindex_like(uniprot_prot1)
    # pho_FC.columns = ["MCF10A at " + str(i) + " min" for i in timepoints]
    pho_FC.columns = time_cols
    pho_FC["R1 Mean"] = uniprot_prot1.mean(axis=1,skipna=True)
    pho_FC["R2 Mean"] = uniprot_prot2.mean(axis=1,skipna=True)
    pho_FC["R1-R2"] = pho_FC["R1 Mean"] - pho_FC["R2 Mean"]
    for i in range(len(time_cols)):
        # pho_FC.iloc[:, i] = (uniprot_prot1.iloc[:, i] + uniprot_prot2.iloc[:, i].mutiply(pho_FC["R1/R2"], axis="index"))/2
        pho_FC.iloc[:, i] = (uniprot_prot1.iloc[:, i] + uniprot_prot2.iloc[:, i] + pho_FC["R1-R2"])/2
        for j in range(pho_FC.shape[0]):
            if np.isfinite(uniprot_prot1.iloc[j, i]):
                if not np.isfinite(uniprot_prot2.iloc[j, i]):
                    pho_FC.iloc[j, i] = uniprot_prot1.iloc[j, i]
            else:
                if np.isfinite(uniprot_prot2.iloc[j, i]):
                    if np.isfinite(pho_FC["R1-R2"][j]):
                        scalor = pho_FC["R1-R2"][j]
                    else:
                        scalor = 1.0
                    pho_FC.iloc[j, i] = uniprot_prot2.iloc[j, i] + scalor
                else:
                    pho_FC.iloc[j, i] = np.nan
    for i in reversed(range(uniprot_prot1.shape[1])):
            pho_FC.iloc[:, i] = pho_FC.iloc[:, i] - pho_FC.iloc[:, 0]
    pho_FC = pho_FC.replace([np.inf, -np.inf], np.nan)
    return(pho_FC)
# %% 
def calculate_fc_dr(uniprot_prot1, uniprot_prot2, dose_cols = ["0.0 ng/ml EGF + mAB225", "0.0 ng/ml EGF", "0.03 ng/ml EGF", "0.1 ng/ml EGF", "0.3 ng/ml EGF", "1.0 ng/ml EGF", "3.0 ng/ml EGF", "10.0 ng/ml EGF", "30.0 ng/ml EGF", "100.0 ng/ml EGF"]):
    pho_FC = pd.DataFrame().reindex_like(uniprot_prot1)
    # pho_FC.columns = ["MCF10A with " + str(i) + " EGF" for i in doses]
    pho_FC.columns = dose_cols
    pho_FC["R1 Mean"] = uniprot_prot1.mean(axis=1,skipna=True)
    pho_FC["R2 Mean"] = uniprot_prot2.mean(axis=1,skipna=True)
    pho_FC["R1-R2"] = pho_FC["R1 Mean"] - pho_FC["R2 Mean"]
    for i in range(len(dose_cols)):
        # pho_FC.iloc[:, i] = (uniprot_prot1.iloc[:, i] + uniprot_prot2.iloc[:, i].mutiply(pho_FC["R1/R2"], axis="index"))/2
        pho_FC.iloc[:, i] = (uniprot_prot1.iloc[:, i] + uniprot_prot2.iloc[:, i] + pho_FC["R1-R2"])/2
        for j in range(pho_FC.shape[0]):
            if np.isfinite(uniprot_prot1.iloc[j, i]):
                if not np.isfinite(uniprot_prot2.iloc[j, i]):
                    pho_FC.iloc[j, i] = uniprot_prot1.iloc[j, i]
            else:
                if np.isfinite(uniprot_prot2.iloc[j, i]):
                    if np.isfinite(pho_FC["R1-R2"][j]):
                        scalor = pho_FC["R1-R2"][j]
                    else:
                        scalor = 1.0
                    pho_FC.iloc[j, i] = uniprot_prot2.iloc[j, i] + scalor
                else:
                    pho_FC.iloc[j, i] = np.nan         
    for i in reversed(range(uniprot_prot1.shape[1])):
        if i != 1:
            pho_FC.iloc[:, i] = pho_FC.iloc[:, i] - pho_FC.iloc[:, 1]
    pho_FC.iloc[:, 1] = pho_FC.iloc[:, 1] - pho_FC.iloc[:, 1]
    pho_FC = pho_FC.replace([np.inf, -np.inf], np.nan)
    return(pho_FC)
# %% 
def calculate_fc_single_ts(uniprot_prot, time_cols = ["0 min", "2 min", "4 min", "8 min", "12 min"]):
    pho_FC = uniprot_prot.copy()
    # pho_FC.columns = ["MCF10A at " + str(i) + " min" for i in timepoints]
    pho_FC.columns = time_cols
    for i in reversed(range(pho_FC.shape[1])):
            pho_FC.iloc[:, i] = pho_FC.iloc[:, i] - pho_FC.iloc[:, 0]
    pho_FC = pho_FC.replace([np.inf, -np.inf], np.nan)
    return(pho_FC)
# %% 
def calculate_fc_single_dr(uniprot_prot, dose_cols = ["0.0 ng/ml EGF + mAB225", "0.0 ng/ml EGF", "0.03 ng/ml EGF", "0.1 ng/ml EGF", "0.3 ng/ml EGF", "1.0 ng/ml EGF", "3.0 ng/ml EGF", "10.0 ng/ml EGF", "30.0 ng/ml EGF", "100.0 ng/ml EGF"]):
    pho_FC = uniprot_prot.copy()
    # pho_FC.columns = ["MCF10A with " + str(i) + "ng/ml EGF" for i in doses]
    pho_FC.columns = dose_cols
    for i in reversed(range(pho_FC.shape[1])):
        if i != 1:
            pho_FC.iloc[:, i] = pho_FC.iloc[:, i] - pho_FC.iloc[:, 1]
    pho_FC.iloc[:, 1] = pho_FC.iloc[:, 1] - pho_FC.iloc[:, 1]
    pho_FC = pho_FC.replace([np.inf, -np.inf], np.nan)
    return(pho_FC)



# %% 
def get_fold_changes(df, fc_fun = None, cols = None):
    df = df.replace([np.inf, -np.inf], np.nan)
    # df = df.dropna(how = "all", subset = ts_R1_val_cols)
    df_fc = fc_fun(df, cols)
    return(df_fc)
# %% 
def get_ave_fold_changes(df1, df2, fc_fun = None, cols = None):
    df1 = df1.replace([np.inf, -np.inf], np.nan)
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    df_fc = fc_fun(df1, df2, cols)
    return(df_fc)



# %% 
def calculate_fc_single_ih(uniprot_prot, inhibitors = None, prefix = None):
    pho_FC = uniprot_prot.copy()
    for inhibitor in inhibitors:
        pho_FC[inhibitor + "_FC_1"] = pho_FC[prefix + inhibitor + '+EGF_' + "R1"] - pho_FC[prefix + inhibitor + '_' + "R1"]
        pho_FC[inhibitor + "_FC_2"] = pho_FC[prefix + inhibitor + '+EGF_' + "R2"] - pho_FC[prefix + inhibitor + '_' + "R2"]
        pho_FC[inhibitor + "_FC_3"] = pho_FC[prefix + inhibitor + '+EGF_' + "R2"] - pho_FC[prefix + inhibitor + '_' + "R1"]
        pho_FC[inhibitor + "_FC_4"] = pho_FC[prefix + inhibitor + '+EGF_' + "R1"] - pho_FC[prefix + inhibitor + '_' + "R2"]
        pho_FC[inhibitor + "_FC_mean"] = pho_FC[[inhibitor + "_FC_" + str(i) for i in range(1,5)]].mean(axis = 1)
        pho_FC[inhibitor + "_FC_std"] = pho_FC[[inhibitor + "_FC_" + str(i) for i in range(1,5)]].std(axis = 1)
    pho_FC = pho_FC.replace([np.inf, -np.inf], np.nan)
    return(pho_FC)



#| lines_to_next_cell: 0
# %% 
def quadratic_spline_roots(spl):
    roots = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a+b)/2), spl(b)
        t = np.roots([u+w-2*v, w-u, 2*v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t*(b-a)/2 + (b+a)/2)
    return np.array(roots)



#| lines_to_next_cell: 0
# %% 
def find_interpolation(t,y):
    f = interpolate.InterpolatedUnivariateSpline(t,y)
    return(f)



#| lines_to_next_cell: 0
# %% 
def find_max_and_half_max_DR(d, y):
    f = interpolate.InterpolatedUnivariateSpline(d,y)
    cr_pts = quadratic_spline_roots(f.derivative())
    cr_pts = np.append(cr_pts, (d[0], d[-1]))
    cr_vals = f(cr_pts)
    min_index = np.argmin(cr_vals)
    max_index = np.argmax(cr_vals)
    if np.abs(cr_vals[min_index]) > np.abs(cr_vals[max_index]):
        half_val = cr_vals[min_index]/2
        #lin_half_val = np.log2((1 - 2 ** cr_vals[min_index])/2)
        max_val = cr_vals[min_index]
        max_pt = cr_pts[min_index]
    else:
        half_val = cr_vals[max_index]/2
        #lin_half_val = np.log2((2 ** cr_vals[max_index] - 1)/2)
        max_val = cr_vals[max_index]
        max_pt = cr_pts[max_index]
    f_shifted = interpolate.InterpolatedUnivariateSpline(d, y - half_val)
    #f_shifted_lin = interpolate.InterpolatedUnivariateSpline(t, y - lin_half_val)
    half_pts = f_shifted.roots()
    half_pt = np.min(half_pts)
    #lin_half_pts = f_shifted_lin.roots()
    #lin_half_pt = np.min(lin_half_pts[lin_half_pts < max_pt])
    #return(np.array([max_val, max_pt, half_val, half_pt, lin_half_val, lin_half_pt]))
    return(np.array([max_val, max_pt, half_val, half_pt]))



# %% 
def find_max_and_half_max_DR(d, y):
    f = interpolate.InterpolatedUnivariateSpline(d,y)
    cr_pts = quadratic_spline_roots(f.derivative())
    cr_pts = np.append(cr_pts, (d[0], d[-1]))
    cr_vals = f(cr_pts)
    min_index = np.argmin(cr_vals)
    max_index = np.argmax(cr_vals)
    if np.abs(cr_vals[min_index]) > np.abs(cr_vals[max_index]):
        half_val = cr_vals[min_index]/2
        #lin_half_val = np.log2((1 - 2 ** cr_vals[min_index])/2)
        max_val = cr_vals[min_index]
        max_pt = cr_pts[min_index]
    else:
        half_val = cr_vals[max_index]/2
        #lin_half_val = np.log2((2 ** cr_vals[max_index] - 1)/2)
        max_val = cr_vals[max_index]
        max_pt = cr_pts[max_index]
    f_shifted = interpolate.InterpolatedUnivariateSpline(d, y - half_val)
    #f_shifted_lin = interpolate.InterpolatedUnivariateSpline(t, y - lin_half_val)
    half_pts = f_shifted.roots()
    half_pt = np.min(half_pts)
    #lin_half_pts = f_shifted_lin.roots()
    #lin_half_pt = np.min(lin_half_pts[lin_half_pts < max_pt])
    #return(np.array([max_val, max_pt, half_val, half_pt, lin_half_val, lin_half_pt]))
    return(np.array([max_val, max_pt, half_val, half_pt]))



# %% 
def get_response_points_ts(dfi, x = None, fc_cols = None):
    df = dfi.copy()
    half_maxima = []
    for i in df.index:
        yi = df.loc[i, fc_cols]
        if yi.isnull().any():
            half_maxima.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        else:
            y = yi.to_numpy()
            half_maxima.append(find_max_and_half_max_DR(x,y))
    half_maxima = np.transpose(np.array(half_maxima))
    df["Max response"] = half_maxima[0]
    df["Max response time"] = half_maxima[1]
    df["Half response"] = half_maxima[2]
    df["Half response time"] = half_maxima[3]
    return(df)



# %% 
def get_response_points_dr(dfi, x = None, fc_cols = None):
    df = dfi.copy()
    half_maxima = []
    for i in df.index:
        yi = df.loc[i, fc_cols]
        if yi.isnull().any():
            half_maxima.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        else:
            y = yi.to_numpy()
            half_maxima.append(find_max_and_half_max_DR(x[1:],y[1:]))
    half_maxima = np.transpose(np.array(half_maxima))
    df["Max response"] = half_maxima[0]
    df["Max response dose"] = half_maxima[1]
    df["Half response"] = half_maxima[2]
    df["Half response dose"] = half_maxima[3]
    return(df)


# %% 
def get_response_df(df, val_cols, res_cols):
    df_sig = df[df[val_cols].abs().max(axis=1, skipna = True)>1]
    df_sig["Site"] = df_sig.index.to_list()
    df_sig_res = df_sig[["Site"] + res_cols]
    df_sig_res = df_sig_res.melt(id_vars = "Site")
    df_sig_res.columns = ["Site", "Response point", res_cols[0][-4:]]
    return(df_sig_res)

# %%
def get_reassignment_summary(df, mono_thr = 60, multi_thr = 60, ptm_label = '*', MS_W = 1, LT_W = 1, MP_W = 1, DT_W = 1):
    goldstd = pd.read_pickle("./Gold standard sites.pkl")
    goldstd_sites = list(set(goldstd["Site ID"]))
    psp = pd.read_pickle("phospho_site_plus_human_clustered.pkl")
    ids = list(set(df["UniProt"]))

    # i = 5
    # id = ids[i]
    # id = "A0A0B4J1U4"
    uniprot_ids = []
    multi_phospho = []
    old_ascores = []
    old_pepts = []
    old_sites = []
    old_goldstd = []
    old_psp_ms_scores = []
    old_psp_lt_scores = []
    old_psp_scores = []
    old_mean_psp_ms_scores = []
    old_mean_psp_lt_scores = []
    old_mean_psp_scores = []
    old_exp_scores = []
    old_mean_exp_scores = []
    old_dist_scores = []
    old_mean_dist_scores = []
    old_comb_scores = []
    old_mean_comb_scores = []
    new_pepts = []
    new_sites = []
    new_goldstd = []
    new_psp_ms_scores = []
    new_psp_lt_scores = []
    new_psp_scores = []
    new_mean_psp_ms_scores = []
    new_mean_psp_lt_scores = []
    new_mean_psp_scores = []
    new_exp_scores = []
    new_mean_exp_scores = []
    new_dist_scores = []
    new_mean_dist_scores = []
    new_comb_scores = []
    new_mean_comb_scores = []
    for idi in ids:
        dfi = df[df["UniProt"] == idi].copy()
        prot_name = dfi["Protein"].to_list()[0]
        pepts = dfi["Peptide"].to_list()
        reses = dfi["Residue"].to_list()
        dfi_ascore = dfi["AScore"].to_list()
        poses = get_res_pos(reses)
        dfi["PhosphoNum"] = [len(res) for res in poses]
        bad_dfi = dfi[(dfi["AScore"] < mono_thr) | ((dfi["AScore"] < multi_thr) & (dfi["PhosphoNum"] > 1))].copy()
        if bad_dfi.shape[0] > 0:
            good_dfi = dfi[(dfi["AScore"] >= multi_thr) & (dfi["PhosphoNum"] <= 1)].copy()
            goodi = get_phosphositeplus_pos(good_dfi["Residue"])
            bad_pepts = bad_dfi["Peptide"].to_list()
            bad_reses = bad_dfi["Residue"].to_list()
            bad_ascores = bad_dfi["AScore"].to_list()
            bad_poses = get_res_pos(bad_reses)
            bad_names = get_res_names(bad_reses)
            pspi = psp[psp["UniProt"] == idi].copy()
            if pspi.shape[0] > 0:
                pspevi = np.array([list(pspi["MS_LIT"].fillna(0) + pspi["MS_CST"].fillna(0)), list(pspi["LT_LIT"].fillna(0))]).transpose()
                psppos = get_phosphositeplus_pos(pspi["MOD_RSD"])
                pspthis = [1 if i in goodi else 0 for i in psppos]
                for i, res in enumerate(bad_poses):
                    ptm_info = get_ptm_info(bad_pepts[i], bad_reses[i])
                    possible_ptm_pos = [ptm[0] for ptm in ptm_info]
                    possible_ptm_scores = []
                    possible_exp_scores = []
                    for ptm_pos in possible_ptm_pos:
                        if ptm_pos in psppos:
                            possible_ptm_scores.append(pspevi[psppos.index(ptm_pos)])
                            possible_exp_scores.append(pspthis[psppos.index(ptm_pos)])
                        elif ptm_pos in goodi:
                            possible_ptm_scores.append(np.zeros(2))
                            possible_exp_scores.append(1)
                        else:
                            possible_ptm_scores.append(np.zeros(2))
                            possible_exp_scores.append(0)
                    possible_ptm_scores /= (np.array(possible_ptm_scores).sum(axis=0, keepdims=True) + 1e-12)
                    possible_exp_scores = np.array(possible_exp_scores)
                    true_res_pos = get_ptm_pos_in_pept(bad_pepts[i], ptm_label)
                    ptm_dists = np.array([np.abs(ptm[2] - np.array(true_res_pos)).min() for ptm in ptm_info])
                    ptm_dist_scores = bad_ascores[i] * (1 - (ptm_dists/ (ptm_dists.max() + 1)))
                    ptm_dist_scores /= (ptm_dist_scores.sum() + 1e-12)
                    ptm_final_scores = possible_ptm_scores[:,0] * MS_W + possible_ptm_scores[:,1] * LT_W + possible_exp_scores * MP_W + ptm_dist_scores * DT_W
                    num_ptm = len(res)
                    if ptm_final_scores.sum() == 0.0:
                        # print(bad_pepts[i] + " does not have PSP sites!")
                        continue
                    new_ptm_indices = np.sort(np.argpartition(ptm_final_scores, -num_ptm)[-num_ptm:])
                    new_ptm_info = [ptm_info[ind] for ind in new_ptm_indices]
                    new_residue = []
                    new_res_pos = []
                    for new_ptm in new_ptm_info:
                        new_residue.append(new_ptm[1] + str(new_ptm[0]) + new_ptm[1].lower())
                        new_res_pos.append(new_ptm[2])
                    new_peptide = relable_pept(bad_pepts[i], sorted(new_res_pos), ptm_label)
                    new_psp_ms_score = [possible_ptm_scores[k][0] for k in new_ptm_indices]
                    new_psp_ms_scores += new_psp_ms_score 
                    new_mean_psp_ms_scores += [np.mean(new_psp_ms_score)] * num_ptm
                    new_psp_lt_score = [possible_ptm_scores[k][1] for k in new_ptm_indices]
                    new_psp_lt_scores += new_psp_lt_score 
                    new_mean_psp_lt_scores += [np.mean(new_psp_lt_score)] * num_ptm
                    new_psp_score = [possible_ptm_scores[k][0] * MS_W + possible_ptm_scores[k][1] * LT_W for k in new_ptm_indices]
                    new_psp_scores += new_psp_score 
                    new_mean_psp_scores += [np.mean(new_psp_score)] * num_ptm
                    new_exp_score = [possible_exp_scores[k] for k in new_ptm_indices]
                    new_exp_scores += new_exp_score
                    new_mean_exp_scores += [np.mean(new_exp_score)] * num_ptm
                    new_dist_score = [ptm_dist_scores[k] for k in new_ptm_indices]
                    new_dist_scores += new_dist_score
                    new_mean_dist_scores += [np.mean(new_dist_score)] * num_ptm
                    new_comb_score = [ptm_final_scores[k] for k in new_ptm_indices]
                    new_comb_scores += new_comb_score
                    new_mean_comb_scores += [np.mean(new_comb_score)] * num_ptm
                    old_ascores += [bad_ascores[i]] * num_ptm
                    old_psp_ms_score = [possible_ptm_scores[possible_ptm_pos.index(res[j])][0] for j in range(num_ptm)]
                    old_psp_ms_scores += old_psp_ms_score
                    old_mean_psp_ms_scores += [np.mean(old_psp_ms_score)] * num_ptm
                    old_psp_lt_score = [possible_ptm_scores[possible_ptm_pos.index(res[j])][1] for j in range(num_ptm)]
                    old_psp_lt_scores += old_psp_lt_score
                    old_mean_psp_lt_scores += [np.mean(old_psp_lt_score)] * num_ptm
                    old_psp_score = [possible_ptm_scores[possible_ptm_pos.index(res[j])][0] * MS_W + possible_ptm_scores[possible_ptm_pos.index(res[j])][1] * LT_W for j in range(num_ptm)]
                    old_psp_scores += old_psp_score
                    old_mean_psp_scores += [np.mean(old_psp_score)] * num_ptm
                    old_exp_score = [possible_exp_scores[possible_ptm_pos.index(res[j])] for j in range(num_ptm)]
                    old_exp_scores += old_exp_score
                    old_mean_exp_scores += [np.mean(old_exp_score)] * num_ptm
                    old_dist_score = [ptm_dist_scores[possible_ptm_pos.index(res[j])] for j in range(num_ptm)]
                    old_dist_scores += old_dist_score
                    old_mean_dist_scores += [np.mean(old_dist_score)] * num_ptm
                    old_comb_score = [possible_ptm_scores[possible_ptm_pos.index(res[j])][0] * MS_W + possible_ptm_scores[possible_ptm_pos.index(res[j])][1] * LT_W + possible_exp_scores[possible_ptm_pos.index(res[j])] * MP_W + ptm_dist_scores[possible_ptm_pos.index(res[j])] * DT_W for j in range(num_ptm)]
                    old_comb_scores += old_comb_score
                    old_mean_comb_scores += [np.mean(old_comb_score)] * num_ptm
                    uniprot_ids += [idi] * num_ptm
                    for j in range(num_ptm):
                        if num_ptm > 1:
                            multi_phospho.append(True)
                        else:
                            multi_phospho.append(False)
                        old_site = prot_name + '-' + bad_names[i][j]
                        old_sites.append(old_site)
                        if old_site in goldstd_sites:
                            old_goldstd.append(True)
                        else:
                            old_goldstd.append(False)
                        old_pepts.append(pepts[i])
                        new_pepts.append(new_peptide)
                        new_site = prot_name + '-' + new_residue[j]
                        new_sites.append(new_site)
                        if new_site in goldstd_sites:
                            new_goldstd.append(True)
                        else:
                            new_goldstd.append(False)
            # else:
                # print(idi + " is not in PhosphoSitePlus database!")
    reassignment_summary = pd.DataFrame({
        "UniProt": uniprot_ids,
        "OldSite": old_sites,
        "NewSite": new_sites,
        "OldPeptide": old_pepts,
        "NewPeptide": new_pepts,
        "MultiPhospho": multi_phospho,
        "OldAScore": old_ascores,
        "OldGoldstd": old_goldstd,
        "NewGoldstd": new_goldstd,
        "OldPspMsScore": old_psp_ms_scores,
        "NewPspMsScore": new_psp_ms_scores,
        "OldPspLtScore": old_psp_lt_scores,
        "NewPspLtScore": new_psp_lt_scores,
        "OldPspScore": old_psp_scores,
        "NewPspScore": new_psp_scores,
        "OldExpScore": old_exp_scores,
        "NewExpScore": new_exp_scores,
        "OldDistScore": old_dist_scores,
        "NewDistScore": new_dist_scores,
        "OldCombScore": old_comb_scores,
        "NewCombScore": new_comb_scores,
        "OldMeanPspMsScore": old_mean_psp_ms_scores,
        "NewMeanPspMsScore": new_mean_psp_ms_scores,
        "OldMeanPspLtScore": old_mean_psp_lt_scores,
        "NewMeanPspLtScore": new_mean_psp_lt_scores,
        "OldMeanPspScore": old_mean_psp_scores,
        "NewMeanPspScore": new_mean_psp_scores,
        "OldMeanExpScore": old_mean_exp_scores,
        "NewMeanExpScore": new_mean_exp_scores,
        "OldMeanDistScore": old_mean_dist_scores,
        "NewMeanDistScore": new_mean_dist_scores,
        "OldMeanCombScore": old_mean_comb_scores,
        "NewMeanCombScore": new_mean_comb_scores,
    })
    return(reassignment_summary)